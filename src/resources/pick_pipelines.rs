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
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[pick_vertex_layout],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[pick_vertex_layout],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[pick_vertex_layout],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

    /// Build the curve POLY_NODE pick pipeline (writes the nearer segment endpoint
    /// node index into the primitive channel). No-op without SHADER_PRIMITIVE_INDEX
    /// or if already built. Group 2 is the per-triangle node payload storage buffer.
    pub(crate) fn ensure_pick_node_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.node_pipeline.is_some() {
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

        let node_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("pick_node_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "pick_node_pipeline_layout",
            &[
                self.pick.camera_bgl.as_ref().expect("pick camera bgl"),
                self.pick
                    .bind_group_layout_1
                    .as_ref()
                    .expect("pick instance bgl"),
                &node_bgl,
            ],
        );

        let shader = crate::resources::builders::wgsl_module(
            device,
            "pick_node_shader",
            crate::resources::builders::with_primitive_index_enable(
                crate::resources::builders::wgsl_source!("pick_node"),
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
                label: "pick_node_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[pick_vertex_layout],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

        self.pick.node_bgl = Some(node_bgl);
        self.pick.node_pipeline = Some(pipeline);
    }

    /// Group 1 layout for the glyph and tensor glyph pick pipelines: the set's
    /// uniform (binding 0) plus the object-id uniform (binding 3). Built once and
    /// shared by both pipelines.
    fn ensure_glyph_pick_id_bgl(&mut self, device: &crate::gpu::Device) {
        if self.pick.glyph_pick_id_bgl.is_some() {
            return;
        }
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("glyph_pick_id_bgl"),
            entries: &[
                // binding 0: the set's glyph / tensor uniform (model + params).
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
                // binding 3: object id to write for this set.
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
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
        self.pick.glyph_pick_id_bgl = Some(bgl);
    }

    /// Lazily create the glyph pick pipeline. Reuses the render glyph vertex
    /// transform (same instance buffer + uniform) with a fragment that writes the
    /// set's object id.
    pub(crate) fn ensure_glyph_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.glyph_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        self.ensure_glyph_pipeline(device);
        self.ensure_glyph_pick_id_bgl(device);

        let camera_bgl = self.pick.camera_bgl.as_ref().expect("pick camera bgl");
        let id_bgl = self
            .pick
            .glyph_pick_id_bgl
            .as_ref()
            .expect("glyph pick id bgl");
        let instance_bgl = self
            .glyph
            .instance_bgl
            .as_ref()
            .expect("glyph instance bgl");

        let shader = crate::resources::builders::wgsl_module(
            device,
            "glyph_pick_shader",
            crate::resources::builders::wgsl_source!("glyph_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "glyph_pick_pipeline_layout",
            &[camera_bgl, id_bgl, instance_bgl],
        );
        let pipeline = build_glyph_pick_pipeline(device, "glyph_pick_pipeline", &layout, &shader);
        self.pick.glyph_pipeline = Some(pipeline);
    }

    /// Lazily create the tensor glyph pick pipeline. Reuses the render tensor
    /// glyph vertex transform (same instance buffer + uniform) with a fragment
    /// that writes the set's object id.
    pub(crate) fn ensure_tensor_glyph_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.tensor_glyph_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        self.ensure_tensor_glyph_pipeline(device);
        self.ensure_glyph_pick_id_bgl(device);

        let camera_bgl = self.pick.camera_bgl.as_ref().expect("pick camera bgl");
        let id_bgl = self
            .pick
            .glyph_pick_id_bgl
            .as_ref()
            .expect("glyph pick id bgl");
        let instance_bgl = self
            .tensor_glyph
            .instance_bgl
            .as_ref()
            .expect("tensor glyph instance bgl");

        let shader = crate::resources::builders::wgsl_module(
            device,
            "tensor_glyph_pick_shader",
            crate::resources::builders::wgsl_source!("tensor_glyph_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "tensor_glyph_pick_pipeline_layout",
            &[camera_bgl, id_bgl, instance_bgl],
        );
        let pipeline =
            build_glyph_pick_pipeline(device, "tensor_glyph_pick_pipeline", &layout, &shader);
        self.pick.tensor_glyph_pipeline = Some(pipeline);
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
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

    /// Lazily create the polyline pick pipeline. The vertex stage copies the
    /// polyline render expansion, reading the same 112-byte per-segment instance
    /// buffer, so the picked ribbon is exactly the screen-space thick line the
    /// render path draws (the viewport size driving the expansion lives in the
    /// polyline uniform, group 1). Group 0 is the minimal pick camera layout;
    /// group 2 is a per-draw object-id uniform.
    pub(crate) fn ensure_polyline_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.polyline_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        self.ensure_polyline_pipeline(device);

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("polyline_pick_id_bgl"),
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

        let camera_bgl = self
            .pick
            .camera_bgl
            .as_ref()
            .expect("ensure_pick_pipeline built the pick camera layout");
        let polyline_bgl = self
            .polyline
            .bgl
            .as_ref()
            .expect("ensure_polyline_pipeline built the polyline group-1 layout");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "polyline_pick_shader",
            crate::resources::builders::wgsl_source!("polyline_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "polyline_pick_pipeline_layout",
            &[camera_bgl, polyline_bgl, &pick_id_bgl],
        );

        // Same 112-byte per-segment instance layout as the polyline render pipeline.
        let attrs = [
            (0u64, 0u32, crate::gpu::VertexFormat::Float32x3),
            (12, 1, crate::gpu::VertexFormat::Float32x3),
            (24, 2, crate::gpu::VertexFormat::Float32x3),
            (36, 3, crate::gpu::VertexFormat::Float32x3),
            (48, 4, crate::gpu::VertexFormat::Float32),
            (52, 5, crate::gpu::VertexFormat::Float32),
            (56, 6, crate::gpu::VertexFormat::Uint32),
            (60, 7, crate::gpu::VertexFormat::Uint32),
            (64, 8, crate::gpu::VertexFormat::Float32x4),
            (80, 9, crate::gpu::VertexFormat::Float32x4),
            (96, 10, crate::gpu::VertexFormat::Float32),
            (100, 11, crate::gpu::VertexFormat::Float32),
            (104, 12, crate::gpu::VertexFormat::Uint32),
        ];
        let vert_attrs: Vec<crate::gpu::VertexAttribute> = attrs
            .iter()
            .map(
                |&(offset, shader_location, format)| crate::gpu::VertexAttribute {
                    offset,
                    shader_location,
                    format,
                },
            )
            .collect();
        let vertex_buffers = [crate::gpu::VertexBufferLayout {
            array_stride: 112,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &vert_attrs,
        }];

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "polyline_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

        self.pick.polyline_pick_id_bgl = Some(pick_id_bgl);
        self.pick.polyline_pipeline = Some(pipeline);
    }

    /// Build the voxel-volume pick pipeline: rasterise the volume bounding cube
    /// and raymarch to the first in-threshold voxel, writing the item's object id
    /// and that voxel's depth. Group 0 is the minimal pick camera (camera +
    /// clip-volume, matching `volume_pick.wgsl`), group 1 reuses the volume
    /// render layout, group 2 is a per-item object-id uniform. No-op if the
    /// volume render layout has not been built yet (no volumes prepared).
    pub(crate) fn ensure_volume_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.volume_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        if self.volume.bgl.is_none() {
            return;
        }

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_pick_id_bgl"),
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

        // Group 0 is the full scene camera layout (camera + clip volume, both
        // FRAGMENT-visible here), not the minimal pick camera: the volume pick
        // fragment reads `view_proj` for the hit depth and the clip volume.
        let camera_bgl = &self.binds.camera_bgl;
        let volume_bgl = self.volume.bgl.as_ref().expect("checked is_some above");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "volume_pick_shader",
            crate::resources::builders::wgsl_source!("volume_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "volume_pick_pipeline_layout",
            &[camera_bgl, volume_bgl, &pick_id_bgl],
        );

        // Position-only unit-cube vertex buffer, matching the volume render cube.
        let vert_layout = crate::gpu::VertexBufferLayout {
            array_stride: 12,
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
                label: "volume_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vert_layout],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_pick"),
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
                // Draw both cube faces (like the volume render): the eye ray march is
                // identical from either face, and back faces keep the volume pickable
                // when the camera is inside the box.
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    front_face: crate::gpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                // The fragment writes `frag_depth` at the hit voxel, so the volume
                // occludes and is occluded by other pick geometry per voxel.
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

        self.pick.volume_pick_id_bgl = Some(pick_id_bgl);
        self.pick.volume_pipeline = Some(pipeline);
    }

    /// Build the GPU implicit-surface pick pipeline: raymarch the SDF isosurface
    /// on a full-screen quad and write the item's object id + hit depth. Group 0
    /// is the full scene camera layout (the fragment reconstructs the ray from
    /// `inv_view_proj`), group 1 reuses the implicit render uniform layout, group 2
    /// is a per-item object-id uniform. No-op if the implicit render layout has not
    /// been built yet (no implicit items prepared).
    pub(crate) fn ensure_implicit_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.implicit_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        if self.implicit.bgl.is_none() {
            return;
        }

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("implicit_pick_id_bgl"),
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

        // Group 0 is the full scene camera layout: the fragment reads
        // `inv_view_proj` and `view_proj` to reconstruct and depth-project the ray.
        let camera_bgl = &self.binds.camera_bgl;
        let implicit_bgl = self.implicit.bgl.as_ref().expect("checked is_some above");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "implicit_pick_shader",
            crate::resources::builders::wgsl_source!("implicit_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "implicit_pick_pipeline_layout",
            &[camera_bgl, implicit_bgl, &pick_id_bgl],
        );

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "implicit_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(crate::gpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_pick"),
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
                // The fragment writes `frag_depth` at the hit point, so the surface
                // occludes and is occluded by other pick geometry per pixel.
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

        self.pick.implicit_pick_id_bgl = Some(pick_id_bgl);
        self.pick.implicit_pipeline = Some(pipeline);
    }

    /// Build the GPU marching-cubes pick pipeline: rasterise the generated MC
    /// vertex buffer (24-byte position + normal, world space) and write the job's
    /// object id + depth. Group 0 is the shared minimal pick camera, group 1 is a
    /// per-job object-id uniform. The draw reuses the render path's indirect args.
    pub(crate) fn ensure_mc_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.mc_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("mc_pick_id_bgl"),
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

        let camera_bgl = self
            .pick
            .camera_bgl
            .as_ref()
            .expect("ensure_pick_pipeline built the pick camera layout");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "mc_pick_shader",
            crate::resources::builders::wgsl_source!("mc_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "mc_pick_pipeline_layout",
            &[camera_bgl, &pick_id_bgl],
        );

        // 24-byte MC vertex: position at offset 0 (normal at 12 is unused here).
        let vert_layout = crate::gpu::VertexBufferLayout {
            array_stride: 24,
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
                label: "mc_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vert_layout],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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
                // The MC surface is generated with a consistent winding but drawn
                // two-sided in the render path; match that so the pick sees both
                // faces.
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

        self.pick.mc_pick_id_bgl = Some(pick_id_bgl);
        self.pick.mc_pipeline = Some(pipeline);
    }

    /// Build the point cloud pick pipeline: reuse the render screen-space quad
    /// expansion (position buffer + `PointCloudUniform` + radius buffer, group 1
    /// reused unchanged from the render bind group) and write the item's object
    /// id plus the hit point's instance index. Group 0 is the full scene camera
    /// layout (the expansion needs the viewport size carried in clip planes),
    /// group 2 is a per-item object-id uniform. No-op if the point cloud render
    /// layout has not been built yet (no point clouds prepared).
    pub(crate) fn ensure_point_cloud_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.point_cloud_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        if self.point_cloud.bgl.is_none() {
            return;
        }

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_pick_id_bgl"),
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

        let camera_bgl = &self.binds.camera_bgl;
        let pc_bgl = self
            .point_cloud
            .bgl
            .as_ref()
            .expect("checked is_some above");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "point_cloud_pick_shader",
            crate::resources::builders::wgsl_source!("point_cloud_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "point_cloud_pick_pipeline_layout",
            &[camera_bgl, pc_bgl, &pick_id_bgl],
        );

        // Same position-per-instance vertex buffer as the point cloud render
        // pipeline.
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
                label: "point_cloud_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

        self.pick.point_cloud_pick_id_bgl = Some(pick_id_bgl);
        self.pick.point_cloud_pipeline = Some(pipeline);
    }

    /// Build the Gaussian splat pick pipeline: reuse the render covariance
    /// projection (`SplatUniform` + sorted-index / position / scale / rotation
    /// storage buffers, group 1 reused unchanged from the render bind group) and
    /// write the item's object id plus the hit splat's instance index. Group 0
    /// is the minimal pick camera (the fragment needs no camera access), group 2
    /// is a per-item object-id uniform. Depth test/write is enabled (unlike the
    /// render pipeline's blend-only depth), so occlusion between splats is
    /// resolved per pixel regardless of draw order. No-op if the splat render
    /// layout has not been built yet (no splat sets prepared).
    pub(crate) fn ensure_gaussian_splat_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.gaussian_splat_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        if self.gaussian_splat.bgl.is_none() {
            return;
        }

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_pick_id_bgl"),
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

        let camera_bgl = self
            .pick
            .camera_bgl
            .as_ref()
            .expect("ensure_pick_pipeline built the pick camera layout");
        let splat_bgl = self
            .gaussian_splat
            .bgl
            .as_ref()
            .expect("checked is_some above");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "gaussian_splat_pick_shader",
            crate::resources::builders::wgsl_source!("gaussian_splat_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "gaussian_splat_pick_pipeline_layout",
            &[camera_bgl, splat_bgl, &pick_id_bgl],
        );

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "gaussian_splat_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

        self.pick.gaussian_splat_pick_id_bgl = Some(pick_id_bgl);
        self.pick.gaussian_splat_pipeline = Some(pipeline);
    }

    /// Build the image slice pick pipeline: reuse the render quad-from-vertex-
    /// index expansion (`ImageSliceUniform`, group 1 reused unchanged from the
    /// render bind group) and write the item's object id. Object-level: the
    /// primitive channel stays 0. Group 0 is the minimal pick camera, group 2 is
    /// a per-item object-id uniform. No-op if the image slice render layout has
    /// not been built yet (no image slices prepared).
    pub(crate) fn ensure_image_slice_pick_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.pick.image_slice_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        if self.image_slice.bgl.is_none() {
            return;
        }

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_pick_id_bgl"),
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

        let camera_bgl = self
            .pick
            .camera_bgl
            .as_ref()
            .expect("ensure_pick_pipeline built the pick camera layout");
        let slice_bgl = self
            .image_slice
            .bgl
            .as_ref()
            .expect("checked is_some above");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "image_slice_pick_shader",
            crate::resources::builders::wgsl_source!("image_slice_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "image_slice_pick_pipeline_layout",
            &[camera_bgl, slice_bgl, &pick_id_bgl],
        );

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "image_slice_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

        self.pick.image_slice_pick_id_bgl = Some(pick_id_bgl);
        self.pick.image_slice_pipeline = Some(pipeline);
    }

    /// Build the volume surface slice pick pipeline: reuse the render mesh
    /// vertex buffer and `VolumeSurfaceSliceUniform` model matrix (group 1
    /// reused unchanged from the render bind group) and write the item's object
    /// id. Object-level: the primitive channel stays 0. Group 0 is the minimal
    /// pick camera, group 2 is a per-item object-id uniform. No-op if the volume
    /// surface slice render layout has not been built yet.
    pub(crate) fn ensure_volume_surface_slice_pick_pipeline(
        &mut self,
        device: &crate::gpu::Device,
    ) {
        if self.pick.volume_surface_slice_pipeline.is_some() {
            return;
        }
        self.ensure_pick_pipeline(device);
        if self.volume.surface_slice_bgl.is_none() {
            return;
        }

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("volume_surface_slice_pick_id_bgl"),
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

        let camera_bgl = self
            .pick
            .camera_bgl
            .as_ref()
            .expect("ensure_pick_pipeline built the pick camera layout");
        let slice_bgl = self
            .volume
            .surface_slice_bgl
            .as_ref()
            .expect("checked is_some above");
        let shader = crate::resources::builders::wgsl_module(
            device,
            "volume_surface_slice_pick_shader",
            crate::resources::builders::wgsl_source!("volume_surface_slice_pick"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "volume_surface_slice_pick_pipeline_layout",
            &[camera_bgl, slice_bgl, &pick_id_bgl],
        );

        let pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "volume_surface_slice_pick_pipeline",
                layout: &layout,
                vertex: crate::gpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                },
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

        self.pick.volume_surface_slice_pick_id_bgl = Some(pick_id_bgl);
        self.pick.volume_surface_slice_pipeline = Some(pipeline);
    }
}

/// Build a glyph / tensor glyph pick pipeline. Both share the same fragment
/// targets as the surface pick pipeline (R32Uint object id + R32Uint primitive
/// id + R32Float depth) and the same depth-stencil setup; only the shader and
/// layout differ.
fn build_glyph_pick_pipeline(
    device: &crate::gpu::Device,
    label: &str,
    layout: &crate::gpu::PipelineLayout,
    shader: &crate::gpu::ShaderModule,
) -> crate::gpu::RenderPipeline {
    // Reuse the full 64-byte Vertex layout so the glyph base mesh binds as-is and
    // the shader reads position + normal like the render path.
    let vertex_layout = Vertex::buffer_layout();
    crate::resources::builders::render_pipeline(
        device,
        crate::resources::builders::RenderPipelineDesc {
            label,
            layout,
            vertex: crate::gpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: crate::gpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(crate::gpu::FragmentState {
                module: shader,
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
                // Glyphs are viewed from any direction; pick both faces like the
                // surface pick pipeline does.
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
    )
}
