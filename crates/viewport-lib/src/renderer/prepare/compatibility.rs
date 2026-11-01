//! Per-frame prepare on the compatibility tier: the lights header and capped light
//! list of its group 0, each viewport's camera and clip uniforms, and the
//! basic mesh draws for the frame's surfaces.

use super::*;
use crate::resources::basic_mesh::{BASIC_INSTANCE_STRIDE, BasicInstance, BasicMeshKey};

/// The compatibility tier's mesh draws for the frame, built in prepare and issued
/// in the LDR pass: opaque draws first, then translucent ones back to front.
#[derive(Default)]
pub(crate) struct BasicMeshDraws {
    instances: Option<crate::gpu::Buffer>,
    capacity: usize,
    bind_groups: Vec<crate::gpu::BindGroup>,
    draws: Vec<BasicMeshDraw>,
}

struct BasicMeshDraw {
    mesh_id: crate::resources::mesh::mesh_store::MeshId,
    key: BasicMeshKey,
    bind_group: usize,
    instance: u32,
    indices: std::ops::Range<u32>,
    /// The scalar stream's sidecar chunk and byte range.
    scalar: Option<(u32, std::ops::Range<u64>)>,
    /// Draws the mesh's edge index buffer rather than its triangles.
    edges: bool,
}

impl ViewportRenderer {
    /// Rank the frame's lights into the compatibility tier's light list and write it with
    /// its header. This tier's profile binds no shadow maps or environment,
    /// and the debug views are off.
    pub(super) fn prepare_compatibility_lights(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        scene_fx: &SceneEffects<'_>,
    ) {
        let (Some(compat), Some(profile)) = (
            self.resources.compatibility_tier.as_deref(),
            self.resources.group0_profile.as_ref(),
        ) else {
            return;
        };
        let lighting = scene_fx.lighting;
        let plugin_lights = lighting::collect_plugin_lights(
            &self.resources,
            &self.item_type_plugins,
            frame,
            self.plugin_frame_index,
        );
        let ranked = lighting::rank_lights(
            lighting
                .lights
                .iter()
                .chain(frame.scene.lights.iter())
                .chain(plugin_lights.iter()),
            frame,
            profile
                .uniform_light_list
                .unwrap_or(crate::resources::MAX_SCENE_LIGHTS),
        );
        self.last_frustum_culled_lights = ranked.frustum_culled;
        self.last_lights_dropped = ranked.dropped;
        let packed: Vec<crate::resources::SingleLightUniform> = ranked
            .lights
            .iter()
            .map(|(_, l)| {
                lighting::build_single_light_uniform(&lighting::rel_light_source(l, &self.origin))
            })
            .collect();

        let header = lighting::lights_header(
            lighting,
            packed.len() as u32,
            profile,
            &lighting::HeaderExtras::default(),
        );
        queue.write_buffer(
            &compat.lights_header_buf,
            0,
            bytemuck::cast_slice(&[header]),
        );
        if !packed.is_empty() {
            queue.write_buffer(&compat.light_list_buf, 0, bytemuck::cast_slice(&packed));
        }
    }

    /// The viewport's camera and clip uniforms, which its compatibility group 0
    /// binds.
    pub(super) fn prepare_viewport_compatibility(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        if self.resources.compatibility_tier.is_none() {
            return;
        }
        let Some(slot) = self.viewport_slots.get(frame.camera.viewport_index) else {
            return;
        };
        let (clip_uniform, clip_vols_uniform) =
            viewport_interaction::clip_uniforms(frame, viewport_fx, &self.origin);
        queue.write_buffer(
            &slot.clip_planes_buf,
            0,
            bytemuck::cast_slice(&[clip_uniform]),
        );
        queue.write_buffer(
            &slot.clip_volume_buf,
            0,
            bytemuck::cast_slice(&[clip_vols_uniform]),
        );
        // The compatibility tier draws through the LDR path, which keeps the lit
        // term in [0, 1] and has no meter for an automatic exposure.
        let mut camera_uniform = frame
            .camera
            .camera_uniform(&self.frame_clock.current(), &self.origin);
        camera_uniform.exposure = viewport_interaction::frame_exposure(frame);
        queue.write_buffer(&slot.camera_buf, 0, bytemuck::cast_slice(&[camera_uniform]));
    }

    /// The compatibility tier's selection outline: a mask binding per selected
    /// surface, then the mask and edge passes into the viewport's outline
    /// targets, which hold their own depth since the tier has no LDR depth of
    /// its own at prepare time.
    pub(super) fn prepare_compatibility_outlines(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        if self.resources.compatibility_tier.is_none() {
            return;
        }
        let vp_idx = frame.camera.viewport_index;
        let mut old = std::mem::take(
            &mut self.viewport_slots[vp_idx]
                .selection_outlines
                .outline_object_buffers,
        )
        .into_iter()
        .map(|b| b.mask);
        let crate::renderer::SurfaceSubmission::Flat(items) = &frame.scene.surfaces;
        let outlined = |item: &crate::renderer::SceneRenderItem| {
            frame.interaction.outline_selected && item.settings.selected && !item.settings.hidden
        };
        if items.iter().any(outlined) || self.any_plugin_item_selected(frame) {
            self.resources.ensure_compatibility_outline_mask(device);
        }
        let mut buffers = Vec::new();
        if let Some(mask) = self
            .resources
            .compatibility_tier
            .as_deref()
            .and_then(|f| f.outline_mask.as_ref())
        {
            for item in items.iter().filter(|i| outlined(i)) {
                if self.resources.mesh_store.get(item.mesh_id).is_none() {
                    continue;
                }
                let uniform = crate::resources::overlay::highlight::OutlineUniform {
                    model: self.origin.rel_transform(&item.model),
                    colour: [0.0; 4],
                    pixel_offset: 0.0,
                    has_position_override: 0,
                    position_override_base: 0,
                    position_override_len: u32::MAX,
                    deform_flags: 0,
                    deform_table: 0,
                    _deform_pad: [0; 2],
                };
                let binding = crate::resources::overlay::highlight::OutlineBinding::reuse_or_new(
                    old.next(),
                    device,
                    queue,
                    &mask.bgl,
                    uniform,
                    None,
                    "outline_mask_object_bg",
                );
                buffers.push(crate::resources::overlay::highlight::OutlineObjectBuffers {
                    mesh_id: item.mesh_id,
                    two_sided: self
                        .resources
                        .material_or_default(item.material)
                        .is_two_sided(),
                    mask: binding,
                });
            }
        }
        self.viewport_slots[vp_idx]
            .selection_outlines
            .outline_object_buffers = buffers;
        if !self.outline_active(frame) {
            return;
        }
        let ppp = frame.camera.pixels_per_point;
        let size = [
            (frame.camera.viewport_size[0] * ppp).round() as u32,
            (frame.camera.viewport_size[1] * ppp).round() as u32,
        ];
        self.encode_outline_passes(device, queue, frame, size, true, sink);
    }

    /// Build the frame's basic mesh draws: one instance record per drawn item
    /// (or per material range), the group 1 bind groups they need, and the
    /// pipeline variants.
    pub(super) fn prepare_basic_meshes(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        if self.resources.compatibility_tier.is_none() {
            return;
        }
        let SurfaceSubmission::Flat(items) = &frame.scene.surfaces;
        // A wireframe item draws its mesh's edges, extracted on first use as
        // on the full tier.
        let wireframe = |item: &crate::SceneRenderItem| {
            frame.viewport.wireframe_mode || item.settings.wireframe
        };
        for item in items.iter().filter(|i| !i.settings.hidden && wireframe(i)) {
            self.resources.ensure_edge_indices(device, item.mesh_id);
        }
        let Some(compat) = self.resources.compatibility_tier.as_deref() else {
            return;
        };
        let resources = &self.resources;
        let eye = frame.camera.render_camera.eye_position();

        let mut records: Vec<BasicInstance> = Vec::new();
        let mut opaque: Vec<BasicMeshDraw> = Vec::new();
        let mut blended: Vec<(f32, BasicMeshDraw)> = Vec::new();
        let mut bind_groups: Vec<crate::gpu::BindGroup> = Vec::new();
        let mut bind_group_index: std::collections::HashMap<
            (
                Option<crate::resources::TextureId>,
                Option<crate::resources::ColourmapId>,
            ),
            usize,
        > = std::collections::HashMap::new();

        for item in items.iter() {
            if item.settings.hidden {
                continue;
            }
            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                continue;
            };
            let edges = wireframe(item);
            if edges && mesh.edge_index_buffer.is_none() {
                // No retained indices to extract edges from.
                continue;
            }
            let ranges: Vec<(std::ops::Range<u32>, crate::resources::MaterialId)> = if edges {
                vec![(0..mesh.edge_index_count, item.material)]
            } else {
                match active_submesh_materials(item, mesh) {
                    Some(mats) => mesh
                        .submeshes
                        .iter()
                        .zip(mats)
                        .map(|(r, m)| (r.first_index..r.first_index + r.index_count, *m))
                        .collect(),
                    None => vec![(0..mesh.index_count, item.material)],
                }
            };

            // Per-vertex scalars bind as a vertex stream. Face, corner and
            // half-edge attributes need the per-face draw, which this path
            // does not have; those items draw in their base colour.
            let scalar = item.active_attribute.as_ref().and_then(|attr| {
                use crate::resources::AttributeKind as K;
                if !matches!(attr.kind, K::Vertex | K::Cell | K::Edge) {
                    return None;
                }
                let record = mesh.sidecar.as_ref()?;
                let stream = record.find(
                    crate::resources::mesh::sidecar_pool::StreamKind::Scalar,
                    &attr.name,
                )?;
                (stream.len as usize >= mesh.vertex_count())
                    .then(|| (record.chunk(), record.byte_range(stream)))
            });
            let (scalar_min, scalar_max) = item
                .active_attribute
                .as_ref()
                .and_then(|a| {
                    item.scalar_range
                        .or(mesh.attribute_ranges.get(&a.name).copied())
                })
                .unwrap_or((0.0, 1.0));
            let lut = item
                .colourmap_id
                .filter(|id| id.0 < resources.content.colourmap_views.len());
            // Eye-relative in f64, so the sort holds far from the world origin.
            let centre =
                (item.model.transform_point3(mesh.aabb.centre().as_dvec3()) - eye).as_vec3();

            for (indices, material_id) in ranges {
                let m = resources.material_or_default(material_id);
                let albedo = m
                    .texture_id
                    .filter(|id| resources.content.textures.get(*id).is_some());
                let bind_group = *bind_group_index.entry((albedo, lut)).or_insert_with(|| {
                    let albedo_view = albedo
                        .and_then(|id| resources.content.textures.get(id))
                        .map_or(&compat.fallback_albedo_view, |t| &t.view);
                    let lut_view = lut.map_or(&resources.content.fallback_lut_view, |id| {
                        &resources.content.colourmap_views[id.0]
                    });
                    bind_groups.push(crate::resources::basic_mesh::basic_mesh_bind_group(
                        device,
                        compat,
                        "basic_mesh_bind_group",
                        albedo_view,
                        lut_view,
                    ));
                    bind_groups.len() - 1
                });

                use crate::resources::basic_mesh::basic_flags as f;
                let flag = |on: bool, bit: u32| if on { bit } else { 0 };
                let mask_cutoff = match m.alpha_mode {
                    crate::scene::material::AlphaMode::Mask(c) => Some(c),
                    _ => None,
                };
                let flags = flag(item.settings.unlit, f::UNLIT)
                    | flag(m.is_two_sided(), f::TWO_SIDED)
                    | flag(mask_cutoff.is_some(), f::ALPHA_MASK)
                    | flag(scalar.is_some(), f::HAS_SCALAR)
                    | flag(item.nan_colour.is_some(), f::USE_NAN_COLOUR)
                    | flag(item.settings.ignore_clip, f::IGNORE_CLIP)
                    | flag(edges, f::WIREFRAME);
                let nan_colour = item.nan_colour.map_or(0, |c| {
                    let [r, g, b, a] = c.to_linear_rgba();
                    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
                    byte(r) | (byte(g) << 8) | (byte(b) << 16) | (byte(a) << 24)
                });
                let bc = item
                    .settings
                    .colour
                    .unwrap_or(m.base_colour)
                    .to_linear_rgb();
                let e = m.emissive_nits();
                let instance = records.len() as u32;
                records.push(BasicInstance {
                    model: self.origin.rel_transform(&item.model),
                    base_colour: [bc[0], bc[1], bc[2], item.settings.opacity],
                    emissive: [e[0], e[1], e[2], mask_cutoff.unwrap_or(0.5)],
                    material: [m.metallic, m.roughness, scalar_min, scalar_max],
                    params: [
                        flags,
                        nan_colour,
                        m.emissive_exposure_weight.clamp(0.0, 1.0).to_bits(),
                        0,
                    ],
                });
                // Edge lines draw opaque whatever the item's opacity, as on the
                // full tier: a translucent wireframe item draws its lines.
                let blend = !edges && (item.settings.opacity < 1.0 || m.is_blend());
                let draw = BasicMeshDraw {
                    mesh_id: item.mesh_id,
                    key: BasicMeshKey {
                        blend,
                        two_sided: m.is_two_sided(),
                        scalar: scalar.is_some() && !edges,
                        lines: edges,
                    },
                    bind_group,
                    instance,
                    indices,
                    scalar: scalar.clone().filter(|_| !edges),
                    edges,
                };
                if blend {
                    blended.push((centre.length_squared(), draw));
                } else {
                    opaque.push(draw);
                }
            }
        }

        // Translucent draws composite back to front.
        blended.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut draws = opaque;
        draws.extend(blended.into_iter().map(|(_, d)| d));

        let draws_state = &mut self.basic_mesh_draws;
        if records.len() > draws_state.capacity {
            let capacity = records.len().next_power_of_two();
            draws_state.instances = Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("basic_mesh_instances"),
                size: capacity as u64 * BASIC_INSTANCE_STRIDE,
                usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            draws_state.capacity = capacity;
        }
        if let Some(buffer) = &draws_state.instances
            && !records.is_empty()
        {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&records));
        }
        draws_state.bind_groups = bind_groups;

        let compat = self
            .resources
            .compatibility_tier
            .as_deref_mut()
            .expect("checked above");
        let layouts = [&compat.camera_bgl, &compat.mesh_bgl];
        for draw in &draws {
            compat.mesh_pipelines.ensure(
                device,
                &self.resources.capabilities,
                layouts,
                self.resources.target_format,
                draw.key,
            );
        }
        self.basic_mesh_draws.draws = draws;
    }

    /// The compatibility tier's scene content in the LDR scene pass: the basic mesh
    /// draws, then the item types that registered on the tier, in the position
    /// the full tier paints them. `camera_bg` is the viewport's group 0.
    pub(crate) fn paint_compatibility_scene(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        camera_bg: &crate::gpu::BindGroup,
        frame: &FrameData,
    ) {
        self.paint_basic_meshes(pass, camera_bg);
        if !self.item_type_plugins.is_empty() {
            pass.set_bind_group(0, camera_bg, &[]);
            self.dispatch_plugin_paint(pass, frame, false);
        }
    }

    /// Issue the basic mesh draws into the LDR scene pass. `camera_bg` is the
    /// viewport's group 0.
    fn paint_basic_meshes(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        camera_bg: &crate::gpu::BindGroup,
    ) {
        let (Some(compat), Some(instances)) = (
            self.resources.compatibility_tier.as_deref(),
            self.basic_mesh_draws.instances.as_ref(),
        ) else {
            return;
        };
        pass.set_bind_group(0, camera_bg, &[]);
        for draw in &self.basic_mesh_draws.draws {
            let (Some(pipeline), Some(mesh)) = (
                compat.mesh_pipelines.get(draw.key),
                self.resources.mesh_store.get(draw.mesh_id),
            ) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &self.basic_mesh_draws.bind_groups[draw.bind_group], &[]);
            pass.set_vertex_buffer(0, self.resources.geometry.vertex_slice(mesh.vertex_span));
            let offset = draw.instance as u64 * BASIC_INSTANCE_STRIDE;
            pass.set_vertex_buffer(1, instances.slice(offset..offset + BASIC_INSTANCE_STRIDE));
            if let Some((chunk, range)) = &draw.scalar {
                pass.set_vertex_buffer(
                    2,
                    self.resources
                        .sidecar
                        .chunk_buffer(*chunk)
                        .slice(range.clone()),
                );
            }
            let indices = match (&mesh.edge_index_buffer, draw.edges) {
                (Some(edges), true) => edges.slice(..),
                (None, true) => continue,
                (_, false) => self.resources.geometry.index_slice(mesh.index_span),
            };
            pass.set_index_buffer(indices, crate::gpu::IndexFormat::Uint32);
            pass.draw_indexed(draw.indices.clone(), 0, 0..1);
        }
    }
}
