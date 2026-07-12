//! GPU object-ID picking: render the scene to an offscreen R32Uint texture
//! and read back the single pixel under the cursor.

use super::*;

/// Item types the GPU pick pass can draw with the shared surface pick pipeline.
///
/// Each variant knows which pick masks it can answer. A type contributes draws
/// only when the caller asked for a level it can resolve, so the mask selects
/// which geometry is rasterised rather than filtering the read-back id. Types
/// with their own pick pipeline (glyphs, sprites, polylines) are handled in
/// their own pass blocks, not here.
#[derive(Clone, Copy)]
enum PickItemType {
    /// Mesh-backed surfaces: scene surfaces and volume-mesh boundaries, resolved
    /// against `mesh_store`.
    Surface,
    /// Tube-family geometry: streamtubes, tubes, and ribbons. These build an
    /// owned connected mesh each frame into the renderer's tube gpu-data vecs
    /// rather than living in `mesh_store`. Object-level only.
    Curve,
    /// Glyph and tensor-glyph sets: instanced base meshes drawn with a dedicated
    /// pick pipeline that reuses the render vertex transform. Object-level only.
    Glyph,
    /// Sprite sets: camera-facing quads expanded in the vertex shader, drawn with
    /// a dedicated pick pipeline that reuses the render expansion. Object-level.
    Sprite,
    /// Polylines: screen-space thick lines expanded per segment in the vertex
    /// shader, drawn with a dedicated pick pipeline. Object-level.
    Polyline,
}

impl PickItemType {
    /// Whether this type answers any level requested in `mask`. A type is drawn
    /// only when the caller asked for something it can resolve.
    fn satisfies(self, mask: crate::interaction::select::pick_mask::PickMask) -> bool {
        use crate::interaction::select::pick_mask::PickMask;
        match self {
            PickItemType::Surface => mask.intersects(
                PickMask::OBJECT
                    | PickMask::FACE
                    | PickMask::VERTEX
                    | PickMask::EDGE
                    | PickMask::CELL,
            ),
            // Tubes and ribbons are object-level only; they answer the whole
            // object mask plus the segment/strip levels a curve query may ask.
            PickItemType::Curve => {
                mask.intersects(PickMask::OBJECT | PickMask::SEGMENT | PickMask::STRIP)
            }
            // Glyph and sprite sets answer the object mask plus the per-instance
            // level.
            PickItemType::Glyph | PickItemType::Sprite => {
                mask.intersects(PickMask::OBJECT | PickMask::INSTANCE)
            }
            // Polylines are object-level; they answer the whole object mask plus
            // the node/segment/strip levels a curve query may ask.
            PickItemType::Polyline => mask.intersects(
                PickMask::OBJECT | PickMask::POLY_NODE | PickMask::SEGMENT | PickMask::STRIP,
            ),
        }
    }
}

/// One glyph or tensor-glyph set to draw into the pick pass. The group-1 bind
/// group (the set uniform + a per-set object-id uniform) is owned; the pipeline,
/// instance bind group, and mesh buffers are borrowed from prepared state.
struct GlyphPickDraw<'a> {
    pipeline: &'a wgpu::RenderPipeline,
    id_bind_group: wgpu::BindGroup,
    instance_bind_group: &'a wgpu::BindGroup,
    vertex_buffer: &'a wgpu::Buffer,
    index_buffer: &'a wgpu::Buffer,
    index_count: u32,
    instance_count: u32,
}

/// One sprite set to draw into the pick pass. The group-2 pick-id bind group is
/// owned; the sprite bind group and position buffer are borrowed from prepared
/// state. The pipeline and group-0 camera bind group are shared across sets.
struct SpritePickDraw<'a> {
    id_bind_group: wgpu::BindGroup,
    sprite_bind_group: &'a wgpu::BindGroup,
    vertex_buffer: &'a wgpu::Buffer,
    sprite_count: u32,
}

/// One polyline to draw into the pick pass. Same shape as [`SpritePickDraw`]:
/// the group-2 pick-id bind group is owned, the render bind group and segment
/// buffer are borrowed. The pipeline and group-0 pick camera bind group are
/// shared across polylines.
struct PolylinePickDraw<'a> {
    id_bind_group: wgpu::BindGroup,
    render_bind_group: &'a wgpu::BindGroup,
    vertex_buffer: &'a wgpu::Buffer,
    segment_count: u32,
}

/// Geometry source for one surface-pipeline pick draw. Surfaces reference a mesh
/// in `mesh_store`; tube-family items reference the owned per-frame buffers built
/// during prepare.
enum PickGeom<'a> {
    /// A mesh handle resolved against `mesh_store`.
    Mesh(crate::resources::mesh::mesh_store::MeshId),
    /// Direct buffer references for a tube-family connected mesh.
    Tube {
        vertex_buffer: &'a wgpu::Buffer,
        index_buffer: &'a wgpu::Buffer,
        index_count: u32,
    },
}

impl ViewportRenderer {
    // -----------------------------------------------------------------------
    // GPU object-ID picking
    // -----------------------------------------------------------------------

    /// GPU object-ID pick: renders the scene to an offscreen `R32Uint` texture
    /// and reads back the single pixel under `cursor`.
    ///
    /// This is O(1) in mesh complexity : every object is rendered with a flat
    /// `u32` ID, and only one pixel is read back. For triangle-level queries
    /// (barycentric scalar probe, exact world position), use the CPU
    /// [`crate::interaction::query::picking::pick_scene_cpu`] path instead.
    ///
    /// The pipeline is lazily initialized on first call : zero overhead when
    /// this method is never invoked.
    ///
    /// # Arguments
    /// * `device` : wgpu device
    /// * `queue` : wgpu queue
    /// * `cursor` : cursor position in viewport-local pixels (top-left origin)
    /// * `frame` : current grouped frame data (camera, scene surfaces, viewport size)
    ///
    /// # Returns
    /// `Some(GpuPickHit)` if an object is under the cursor, `None` if empty space.
    pub fn pick_scene_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
    ) -> Option<crate::interaction::query::picking::GpuPickHit> {
        self.pick_scene_gpu_masked(
            device,
            queue,
            cursor,
            frame,
            crate::interaction::select::pick_mask::PickMask::all(),
        )
    }

    /// GPU object-ID pick restricted to the item types `mask` selects.
    ///
    /// A type is drawn into the pick pass only when it answers a bit in `mask`,
    /// so a typed query (say an instance-only mask) is never occluded by an
    /// object of a type the caller did not ask for. Types with no pick pipeline
    /// yet do not draw, so they read back as no hit rather than a wrong hit.
    pub(crate) fn pick_scene_gpu_masked(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
        mask: crate::interaction::select::pick_mask::PickMask,
    ) -> Option<crate::interaction::query::picking::GpuPickHit> {
        // In Playback mode, throttle picking to every 4th frame to reduce overhead
        // during animation. Interactive, Paused, and Capture modes always pick.
        if self.runtime_mode == crate::renderer::stats::RuntimeMode::Playback
            && self.frame_counter % 4 != 0
        {
            return None;
        }

        // Read scene items from the surface submission.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        let ppp = frame.camera.pixels_per_point;
        let vp_w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let vp_h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        // --- bounds check (logical coordinates match the logical cursor) ---
        if cursor.x < 0.0
            || cursor.y < 0.0
            || cursor.x >= frame.camera.viewport_size[0]
            || cursor.y >= frame.camera.viewport_size[1]
            || vp_w == 0
            || vp_h == 0
        {
            return None;
        }

        // --- lazy pipeline init ---
        self.resources.ensure_pick_pipeline(device);
        let glyph_wanted = PickItemType::Glyph.satisfies(mask);
        let has_pickable_glyphs = glyph_wanted
            && self
                .glyph_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.instance_count > 0);
        let has_pickable_tensor = glyph_wanted
            && self
                .tensor_glyph_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.instance_count > 0);
        if has_pickable_glyphs {
            self.resources.ensure_glyph_pick_pipeline(device);
        }
        if has_pickable_tensor {
            self.resources.ensure_tensor_glyph_pick_pipeline(device);
        }
        let has_pickable_sprites = PickItemType::Sprite.satisfies(mask)
            && self
                .sprite_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.sprite_count > 0);
        if has_pickable_sprites {
            self.resources.ensure_sprite_pick_pipeline(device);
        }
        let has_pickable_polylines = PickItemType::Polyline.satisfies(mask)
            && self
                .polyline_gpu_data
                .iter()
                .any(|g| g.pick_id != PickId::NONE && g.segment_count > 0);
        if has_pickable_polylines {
            self.resources.ensure_polyline_pick_pipeline(device);
        }

        // --- build PickInstance data ---
        // Every mesh-backed pickable item draws through the surface pipeline:
        // scene surfaces, plus volume-mesh boundaries (both opaque and
        // transparent, which render their boundary as a surface mesh). Items that
        // are hidden or have pick_id 0 are skipped; clear value 0 means "no hit".
        let pickable =
            |item: &SceneRenderItem| !item.settings.hidden && item.settings.pick_id != PickId::NONE;
        let to_instance = |item: &SceneRenderItem| {
            let m = item.model;
            PickInstance {
                model_c0: m[0],
                model_c1: m[1],
                model_c2: m[2],
                model_c3: m[3],
                object_id: item.settings.pick_id.0 as u32,
                _pad: [0; 3],
            }
        };

        let instance_from = |model: [[f32; 4]; 4], pick_id: PickId| PickInstance {
            model_c0: model[0],
            model_c1: model[1],
            model_c2: model[2],
            model_c3: model[3],
            object_id: pick_id.0 as u32,
            _pad: [0; 3],
        };

        let mut draws: Vec<(PickGeom, PickInstance)> = Vec::new();

        // Surfaces and volume-mesh boundaries draw through the Surface pipeline;
        // skip building their instance data when the mask asks for none of the
        // levels that type answers.
        if PickItemType::Surface.satisfies(mask) {
            for item in scene_items.iter().filter(|i| pickable(i)) {
                draws.push((PickGeom::Mesh(item.mesh_id), to_instance(item)));
            }
            for ri in frame
                .scene
                .volume_meshes
                .iter()
                .map(|vm| vm.to_render_item())
                .filter(pickable)
            {
                draws.push((PickGeom::Mesh(ri.mesh_id), to_instance(&ri)));
            }
        }

        // Streamtubes, tubes, and ribbons build owned connected meshes into these
        // vecs during prepare(); each entry carries its source item's pick_id and
        // model. The streamtube shader applies the model to the buffer positions,
        // so the pick pass uses the same matrix and its silhouette matches.
        if PickItemType::Curve.satisfies(mask) {
            for family in [
                self.streamtube_gpu_data.as_slice(),
                self.tube_gpu_data.as_slice(),
                self.ribbon_gpu_data.as_slice(),
            ] {
                for gpu in family
                    .iter()
                    .filter(|g| g.pick_id != PickId::NONE && g.index_count > 0)
                {
                    draws.push((
                        PickGeom::Tube {
                            vertex_buffer: &gpu.vertex_buffer,
                            index_buffer: &gpu.index_buffer,
                            index_count: gpu.index_count,
                        },
                        instance_from(gpu.model, gpu.pick_id),
                    ));
                }
            }
        }

        // Glyph and tensor-glyph sets draw with their own pick pipelines. Each
        // set builds a group-1 bind group (the set uniform + a per-set object-id
        // uniform) here so it outlives the render pass. The buffers behind the
        // bind group stay alive through it, so the temporary id buffer can drop.
        let mut glyph_draws: Vec<GlyphPickDraw> = Vec::new();
        if has_pickable_glyphs || has_pickable_tensor {
            let id_bgl = self
                .resources
                .pick
                .glyph_pick_id_bgl
                .as_ref()
                .expect("glyph pick id bgl");
            let make_id_bg = |pick_id: PickId, uniform_buf: &wgpu::Buffer| {
                let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("glyph_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("glyph_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: id_buf.as_entire_binding(),
                        },
                    ],
                })
            };
            if has_pickable_glyphs {
                let pipeline = self
                    .resources
                    .pick
                    .glyph_pipeline
                    .as_ref()
                    .expect("glyph pick pipeline");
                for gpu in self
                    .glyph_gpu_data
                    .iter()
                    .filter(|g| g.pick_id != PickId::NONE && g.instance_count > 0)
                {
                    glyph_draws.push(GlyphPickDraw {
                        pipeline,
                        id_bind_group: make_id_bg(gpu.pick_id, &gpu._uniform_buf),
                        instance_bind_group: &gpu.instance_bind_group,
                        vertex_buffer: gpu.mesh_vertex_buffer,
                        index_buffer: gpu.mesh_index_buffer,
                        index_count: gpu.mesh_index_count,
                        instance_count: gpu.instance_count,
                    });
                }
            }
            if has_pickable_tensor {
                let pipeline = self
                    .resources
                    .pick
                    .tensor_glyph_pipeline
                    .as_ref()
                    .expect("tensor glyph pick pipeline");
                for gpu in self
                    .tensor_glyph_gpu_data
                    .iter()
                    .filter(|g| g.pick_id != PickId::NONE && g.instance_count > 0)
                {
                    glyph_draws.push(GlyphPickDraw {
                        pipeline,
                        id_bind_group: make_id_bg(gpu.pick_id, &gpu._uniform_buf),
                        instance_bind_group: &gpu.instance_bind_group,
                        vertex_buffer: gpu.mesh_vertex_buffer,
                        index_buffer: gpu.mesh_index_buffer,
                        index_count: gpu.mesh_index_count,
                        instance_count: gpu.instance_count,
                    });
                }
            }
        }

        // Sprite sets draw with their own pipeline. Each set gets a group-2 bind
        // group holding its object id; the pipeline and camera bind group are
        // shared, so only the id, sprite bind group, and position buffer vary.
        let mut sprite_draws: Vec<SpritePickDraw> = Vec::new();
        if has_pickable_sprites {
            let id_bgl = self
                .resources
                .sprite
                .pick_id_bgl
                .as_ref()
                .expect("sprite pick id bgl");
            for gpu in self
                .sprite_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE && g.sprite_count > 0)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sprite_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sprite_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                sprite_draws.push(SpritePickDraw {
                    id_bind_group,
                    sprite_bind_group: &gpu.bind_group,
                    vertex_buffer: &gpu.vertex_buffer,
                    sprite_count: gpu.sprite_count,
                });
            }
        }

        // Polylines draw with their own pipeline: group 0 is the shared minimal
        // pick camera, group 1 is the polyline render bind group (uniform + LUT),
        // group 2 is the per-draw object id.
        let mut polyline_draws: Vec<PolylinePickDraw> = Vec::new();
        if has_pickable_polylines {
            let id_bgl = self
                .resources
                .pick
                .polyline_pick_id_bgl
                .as_ref()
                .expect("polyline pick id bgl");
            for gpu in self
                .polyline_gpu_data
                .iter()
                .filter(|g| g.pick_id != PickId::NONE && g.segment_count > 0)
            {
                let id_data = [gpu.pick_id.0 as u32, 0u32, 0u32, 0u32];
                let id_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("polyline_pick_id_buf"),
                    size: std::mem::size_of_val(&id_data) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&id_buf, 0, bytemuck::cast_slice(&id_data));
                let id_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("polyline_pick_id_bg"),
                    layout: id_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: id_buf.as_entire_binding(),
                    }],
                });
                polyline_draws.push(PolylinePickDraw {
                    id_bind_group,
                    render_bind_group: &gpu.bind_group,
                    vertex_buffer: &gpu.vertex_buffer,
                    segment_count: gpu.segment_count,
                });
            }
        }

        if draws.is_empty()
            && glyph_draws.is_empty()
            && sprite_draws.is_empty()
            && polyline_draws.is_empty()
        {
            return None;
        }

        let pick_instances: Vec<PickInstance> = draws.iter().map(|(_, inst)| *inst).collect();

        // --- pick instance storage buffer + bind group ---
        let pick_instance_bytes = bytemuck::cast_slice(&pick_instances);
        let pick_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_instance_buf"),
            size: pick_instance_bytes.len().max(80) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_instance_buf, 0, pick_instance_bytes);

        let pick_instance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_instance_bg"),
            layout: self
                .resources
                .pick
                .bind_group_layout_1
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pick_instance_buf.as_entire_binding(),
            }],
        });

        // --- pick camera uniform buffer + bind group ---
        let camera_uniform = frame.camera.render_camera.camera_uniform();
        let camera_bytes = bytemuck::bytes_of(&camera_uniform);
        let pick_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_camera_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_camera_buf, 0, camera_bytes);

        let pick_camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_camera_bg"),
            layout: self
                .resources
                .pick
                .camera_bgl
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.resources.clip_volume_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // --- offscreen pick textures (R32Uint + R32Float) + depth ---
        let pick_id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_id_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_id_view = pick_id_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Primitive-id target. The pick pipelines write a sub-object index here
        // (0 for now); it is attached so the pipeline's location-1 output has a
        // target, but it is not read back until sub-object picking uses it.
        let pick_prim_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_prim_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_prim_view = pick_prim_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let pick_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_depth_colour_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_depth_view =
            pick_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_ds_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_stencil_view =
            depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- render pass ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pick_pass_encoder"),
        });
        {
            let mut pick_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_id_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_prim_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_depth_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_stencil_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Surface-pipeline draws: scene surfaces, volume-mesh boundaries, and
            // tube-family geometry all rasterise with the shared pick pipeline.
            // Type-level mask filtering already happened while building `draws`,
            // so an unbuilt or unrequested type contributes nothing and reads
            // back as no hit. Instance index in the storage buffer = position in
            // `draws`.
            pick_pass.set_pipeline(
                self.resources
                    .pick
                    .pipeline
                    .as_ref()
                    .expect("ensure_pick_pipeline must be called first"),
            );
            pick_pass.set_bind_group(0, &pick_camera_bg, &[]);
            pick_pass.set_bind_group(1, &pick_instance_bg, &[]);

            for (instance_slot, (geom, _)) in draws.iter().enumerate() {
                let slot = instance_slot as u32;
                match geom {
                    PickGeom::Mesh(mesh_id) => {
                        let Some(mesh) = self.resources.mesh_store.get(*mesh_id) else {
                            continue;
                        };
                        pick_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pick_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pick_pass.draw_indexed(0..mesh.index_count, 0, slot..slot + 1);
                    }
                    PickGeom::Tube {
                        vertex_buffer,
                        index_buffer,
                        index_count,
                    } => {
                        pick_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        pick_pass
                            .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pick_pass.draw_indexed(0..*index_count, 0, slot..slot + 1);
                    }
                }
            }

            // Glyph / tensor-glyph sets: each draws its instanced base mesh with a
            // dedicated pipeline that reuses the render vertex transform and writes
            // the set's object id.
            for gd in &glyph_draws {
                pick_pass.set_pipeline(gd.pipeline);
                pick_pass.set_bind_group(0, &pick_camera_bg, &[]);
                pick_pass.set_bind_group(1, &gd.id_bind_group, &[]);
                pick_pass.set_bind_group(2, gd.instance_bind_group, &[]);
                pick_pass.set_vertex_buffer(0, gd.vertex_buffer.slice(..));
                pick_pass.set_index_buffer(gd.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pick_pass.draw_indexed(0..gd.index_count, 0, 0..gd.instance_count);
            }

            // Sprite sets: camera-facing quads expanded in the vertex shader. The
            // pipeline and full camera bind group (group 0) are shared; each set
            // varies its sprite bind group, pick-id, and position buffer.
            if let Some(sprite_pipeline) = self.resources.sprite.pick_pipeline.as_ref() {
                if !sprite_draws.is_empty() {
                    pick_pass.set_pipeline(sprite_pipeline);
                    pick_pass.set_bind_group(0, &self.resources.camera_bind_group, &[]);
                    for sd in &sprite_draws {
                        pick_pass.set_bind_group(1, sd.sprite_bind_group, &[]);
                        pick_pass.set_bind_group(2, &sd.id_bind_group, &[]);
                        pick_pass.set_vertex_buffer(0, sd.vertex_buffer.slice(..));
                        pick_pass.draw(0..6, 0..sd.sprite_count);
                    }
                }
            }

            // Polylines: screen-space thick lines, one draw per polyline of its
            // segment quads. Group 0 is the shared minimal pick camera.
            if let Some(polyline_pipeline) = self.resources.pick.polyline_pipeline.as_ref() {
                if !polyline_draws.is_empty() {
                    pick_pass.set_pipeline(polyline_pipeline);
                    pick_pass.set_bind_group(0, &pick_camera_bg, &[]);
                    for pd in &polyline_draws {
                        pick_pass.set_bind_group(1, pd.render_bind_group, &[]);
                        pick_pass.set_bind_group(2, &pd.id_bind_group, &[]);
                        pick_pass.set_vertex_buffer(0, pd.vertex_buffer.slice(..));
                        pick_pass.draw(0..6, 0..pd.segment_count);
                    }
                }
            }
        }

        // --- copy 1x1 pixels to staging buffers ---
        // R32Uint: 4 bytes per pixel, min bytes_per_row = 256 (wgpu alignment)
        let bytes_per_row_aligned = 256u32; // wgpu requires multiples of 256

        let id_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_id_staging"),
            size: bytes_per_row_aligned as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let depth_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_depth_staging"),
            size: bytes_per_row_aligned as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Convert logical cursor to physical pixel coordinates for the pick texture readback.
        let px = (cursor.x * ppp).round() as u32;
        let py = (cursor.y * ppp).round() as u32;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick_id_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &id_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick_depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &depth_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // --- map and read ---
        let (tx_id, rx_id) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx_dep, rx_dep) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        id_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_id.send(r);
            });
        depth_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_dep.send(r);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let _ = rx_id.recv().unwrap_or(Err(wgpu::BufferAsyncError));
        let _ = rx_dep.recv().unwrap_or(Err(wgpu::BufferAsyncError));

        let object_id = {
            let data = id_staging.slice(..).get_mapped_range();
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        id_staging.unmap();

        let depth = {
            let data = depth_staging.slice(..).get_mapped_range();
            f32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        depth_staging.unmap();

        // 0 = miss (clear colour or non-pickable surface).
        if object_id == 0 {
            return None;
        }

        Some(crate::interaction::query::picking::GpuPickHit {
            object_id: PickId(object_id as u64),
            depth,
        })
    }
}
