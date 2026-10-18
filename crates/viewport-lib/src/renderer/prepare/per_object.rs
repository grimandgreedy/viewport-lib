//! Non-instanced (per-object) mesh draw preparation.

use super::*;

/// Assemble the instance record for a scene item: the model transform, the
/// per-instance flags and the per-mesh sidecar words every mesh draw reads at
/// group 1 binding 0. `mesh` is the item's resident mesh, looked up once by
/// the caller. Shared by every packer so the two draw paths cannot drift.
pub(crate) fn build_instance_record(
    resources: &DeviceResources,
    item: &SceneRenderItem,
    mesh: Option<&crate::resources::GpuMesh>,
    wireframe_mode: bool,
    light_probe_index: Option<u32>,
    material_id: u32,
) -> InstanceData {
    use crate::resources::mesh::instancing::instance_flags as f;
    let m = resources.material_or_default(item.material);
    let resolved = resources.resolve_material_slots(m);
    let cm = common_material(item, m, resolved);
    let mesh_closed = mesh.is_some_and(|m| m.closed);
    let has_position_override = mesh.is_some_and(|m| m.position_override_buffer.is_some());
    let has_normal_override = mesh.is_some_and(|m| m.normal_override_buffer.is_some());
    let record = mesh.and_then(|m| m.sidecar.as_ref());
    // The item's own lightmap placement, else the one its mesh's lightmap was
    // registered with.
    let placement = item
        .lightmap_placement
        .and_then(|id| resources.frame_lightmap_placements.get(id.index()).copied())
        .unwrap_or_else(|| {
            mesh.and_then(|m| m.lightmap.as_ref())
                .map_or(crate::resources::LightmapPlacement::WHOLE, |lm| {
                    crate::resources::LightmapPlacement::new(lm.layer, lm.scale_bias)
                })
        });
    // The streams the vertex stage reads, as first words in the record's
    // chunk (0 for none). The shader reads them without a length, so a stream
    // is only taken when it covers every vertex index the draw reaches: the
    // vertex count for an indexed draw, the corner count for a per-face one.
    // A face, corner or half-edge attribute draws the mesh's per-face
    // expanded vertex buffer, unless the item draws its edges instead.
    let per_face_draw = !(wireframe_mode || item.settings.wireframe)
        && item.active_attribute.as_ref().is_some_and(|a| {
            use crate::resources::AttributeKind as K;
            matches!(a.kind, K::Face | K::FaceColour | K::Halfedge | K::Corner)
        });
    let (colour_stream, warp_stream, lightmap_stream) = {
        use crate::resources::AttributeKind as K;
        use crate::resources::mesh::sidecar_pool::StreamKind;
        let attr = item.active_attribute.as_ref();
        let per_face = attr
            .is_some_and(|a| matches!(a.kind, K::Face | K::FaceColour | K::Halfedge | K::Corner));
        let reach = mesh.map_or(0, |m| {
            if per_face {
                m.index_count
            } else {
                m.vertex_count() as u32
            }
        });
        let word = |s: Option<&crate::resources::mesh::sidecar_pool::SidecarStream>| {
            record
                .zip(s)
                .filter(|(_, s)| s.len >= reach)
                .map_or(0, |(r, s)| r.stream_word(s))
        };
        let find = |kind, name: &str| record.and_then(|r| r.find(kind, name));
        let colour = attr.map_or(0, |a| {
            if a.kind == K::FaceColour {
                word(find(StreamKind::FaceColour, &a.name))
            } else {
                // A scalar name resolves against the streams of its own
                // shape first, then the other shape when that covers the draw.
                let (first, second) = if per_face {
                    (StreamKind::FaceScalar, StreamKind::Scalar)
                } else {
                    (StreamKind::Scalar, StreamKind::FaceScalar)
                };
                match word(find(first, &a.name)) {
                    0 => word(find(second, &a.name)),
                    w => w,
                }
            }
        });
        let warp = item
            .warp_attribute
            .as_deref()
            .map_or(0, |n| word(find(StreamKind::Vector, n)));
        let lightmap = word(record.and_then(|r| r.first_of(StreamKind::LightmapUv)));
        (colour, warp, lightmap)
    };
    // Styled back-face `Pattern` world scale: transform the mesh AABB by this
    // instance's model to get its world extent, then divide the pattern scale
    // by it. Zero for every non-Pattern policy.
    let backface_pattern_scale = match m.backface_policy {
        crate::scene::material::BackfacePolicy::Pattern(cfg) => {
            let world_extent = mesh
                .map(|m| {
                    m.aabb
                        .transformed(&glam::Mat4::from_cols_array_2d(&item.model))
                        .longest_side()
                })
                .unwrap_or(1.0)
                .max(1e-6);
            cfg.scale / world_extent
        }
        _ => 0.0,
    };
    let flags = f::when(cm.selected != 0, f::SELECTED)
        | f::when(wireframe_mode || item.settings.wireframe, f::WIREFRAME)
        | f::when(cm.has_texture != 0, f::HAS_TEXTURE)
        | f::when(cm.has_normal_map != 0, f::HAS_NORMAL_MAP)
        | f::when(cm.has_ao_map != 0, f::HAS_AO_MAP)
        | f::when(cm.unlit != 0, f::UNLIT)
        | f::when(cm.receive_shadows != 0, f::RECEIVE_SHADOWS)
        | f::when(mesh_closed, f::MESH_CLOSED)
        | f::when(m.is_two_sided() && !mesh_closed, f::TWO_SIDED_RECEIVER)
        | f::when(
            matches!(m.alpha_mode, crate::scene::material::AlphaMode::Mask(_)),
            f::ALPHA_MASK,
        )
        | f::when(light_probe_index.is_some(), f::HAS_LIGHT_PROBE)
        | f::when(item.settings.ignore_clip, f::IGNORE_CLIP)
        | f::when(has_position_override, f::HAS_POSITION_OVERRIDE)
        | f::when(has_normal_override, f::HAS_NORMAL_OVERRIDE)
        | f::when(item.active_attribute.is_some(), f::HAS_ATTRIBUTE)
        | f::when(
            item.active_attribute
                .as_ref()
                .is_some_and(|a| a.kind == crate::resources::AttributeKind::FaceColour),
            f::USE_FACE_COLOUR,
        )
        | f::when(item.warp_attribute.is_some(), f::HAS_WARP)
        | f::when(record.is_some(), f::HAS_SIDECAR)
        | f::when(mesh.is_some_and(|m| m.has_uv1) && !per_face_draw, f::HAS_UV1);
    InstanceData {
        model: cm.model,
        colour: cm.colour,
        flags,
        material_id,
        alpha_cutoff: match m.alpha_mode {
            crate::scene::material::AlphaMode::Mask(c) => c,
            _ => 0.5,
        },
        light_probe_index: light_probe_index.unwrap_or(0),
        custom_data_id: 0,
        backface_pattern_scale,
        object_mask: item.settings.visibility_mask,
        // The override windows stay in the mesh's sidecar record, read only
        // under their flags; the streams the item reads are resolved here.
        sidecar_record: record.map_or(0, |r| r.base_word()),
        deform_flags: resources.deform.flag_bits(item.mesh_id),
        colour_stream,
        warp_scale: item.warp_scale,
        lightmap_layer: placement.layer,
        lightmap_scale_bias: placement.scale_bias,
        warp_stream,
        lightmap_stream,
        // A per-face draw reads its own expanded vertex buffer from vertex 0;
        // every other draw binds the whole slab chunk at the mesh's base.
        vertex_base: if per_face_draw {
            0
        } else {
            mesh.map_or(0, |m| resources.geometry.base_vertex(m.vertex_span) as u32)
        },
        _pad: 0,
    }
}

/// Light-probe SH prepass, shared by the per-object and instanced draw paths.
///
/// For each item that opts into a light probe (`indirect_light == LightProbe`),
/// blend the nearby probes' SH at the object position and pack it into the
/// shared SH buffer (group 0 binding 18). The returned vector maps this frame's
/// scene-item index to that object's SH block index; both paths read it so an
/// item's `light_probe_index` is identical whichever path draws it. Returns all
/// `None` (and writes nothing) when no probes are uploaded, so it costs nothing
/// for consumers that never call `set_light_probes`.
pub(super) fn prepare_light_probe_sh(
    resources: &DeviceResources,
    scene_items: &[SceneRenderItem],
    queue: &crate::gpu::Queue,
) -> Vec<Option<u32>> {
    use crate::renderer::IndirectLightSource;
    let mut indices = vec![None; scene_items.len()];
    let probes = resources.lighting.probes.as_ref().filter(|p| !p.is_empty());
    let mut sh_gpu: Vec<[f32; 4]> = Vec::new();
    let mut count = 0u32;
    for (idx, item) in scene_items.iter().enumerate() {
        match item.indirect_light {
            IndirectLightSource::LightProbe => {
                // Blend the nearby point probes' SH at the object position and
                // pack a block; the block index rides light_probe_index.
                if let Some(probes) = probes {
                    if (count as usize) < crate::resources::light_probes::MAX_LIGHT_PROBE_OBJECTS {
                        let center = [item.model[3][0], item.model[3][1], item.model[3][2]];
                        sh_gpu.extend_from_slice(&probes.blend_sh_at(center).to_gpu());
                        indices[idx] = Some(count);
                        count += 1;
                    }
                }
            }
            IndirectLightSource::ProbeVolume => {
                // The volume is sampled per fragment in the shader, so no CPU
                // blend and no SH block: the sentinel index selects it. Works
                // whether or not a point-probe set is also uploaded.
                indices[idx] = Some(crate::resources::light_probes::PROBE_VOLUME_INDEX);
            }
            IndirectLightSource::GlobalIbl => {}
        }
    }
    if !sh_gpu.is_empty() {
        queue.write_buffer(
            &resources.lighting.indirect_buf,
            0,
            bytemuck::cast_slice(&sh_gpu),
        );
    }
    indices
}

impl ViewportRenderer {
    /// Write the per-item uniforms and build the group-1 bind groups for one
    /// viewport's foreground items.
    ///
    /// Foreground items live outside the shared [`PerObjectState`] cache: the
    /// list is small and per-viewport, and running the `(pick_id, occurrence)`
    /// keyed cache over a second list would collide with the scene items'
    /// entries. `entries` is index-aligned with `items`; an entry whose mesh
    /// is missing keeps `bind_group: None` and the draw path skips it.
    pub(super) fn prepare_foreground_objects(
        resources: &mut DeviceResources,
        entries: &mut Vec<crate::renderer::per_object_state::ForegroundObjectEntry>,
        items: &[SceneRenderItem],
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        entries.truncate(items.len());
        let uniform_size = std::mem::size_of::<InstanceData>() as u64;
        for (idx, item) in items.iter().enumerate() {
            if entries.len() <= idx {
                let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("foreground_object_uniform"),
                    size: uniform_size,
                    usage: crate::gpu::BufferUsages::STORAGE
                        | crate::gpu::BufferUsages::VERTEX
                        | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                entries.push(crate::renderer::per_object_state::ForegroundObjectEntry {
                    uniform_buf: buf,
                    bind_group: None,
                    cache_key: 0,
                    last_uniform: None,
                });
            }
            if resources.mesh_store.get(item.mesh_id).is_none() {
                tracing::warn!(
                    mesh_index = item.mesh_id.index(),
                    "foreground item mesh_index invalid, skipping"
                );
                entries[idx].bind_group = None;
                continue;
            }

            let m = *resources.material_or_default(item.material);
            resources.update_mesh_texture_bind_group(
                device,
                item.mesh_id,
                m.texture_id,
                m.normal_map_id,
                m.ao_map_id,
                item.colourmap_id,
                m.matcap_id(),
                m.metallic_roughness_texture_id,
                m.emissive_texture_id,
            );

            let shading = crate::resources::material_gpu::ItemShading::from_item(
                item,
                &m,
                resources.mesh_store.get(item.mesh_id),
            );
            let material_id = resources.material_block(item.material, &shading);
            let mesh = resources.mesh_store.get(item.mesh_id);
            let obj_uniform =
                build_instance_record(resources, item, mesh, false, None, material_id);
            let entry = &mut entries[idx];
            let uniform_changed = entry.last_uniform.as_ref().map_or(true, |u| {
                bytemuck::bytes_of(u) != bytemuck::bytes_of(&obj_uniform)
            });
            if uniform_changed {
                queue.write_buffer(&entry.uniform_buf, 0, bytemuck::cast_slice(&[obj_uniform]));
                entry.last_uniform = Some(obj_uniform);
                resources.frame_upload_bytes += uniform_size;
            }

            let prev_key = entry.bind_group.as_ref().map(|_| entry.cache_key);
            let built = resources.build_per_item_object_bind_group(
                device,
                item.mesh_id,
                &entry.uniform_buf,
                m.texture_id,
                m.normal_map_id,
                m.ao_map_id,
                item.colourmap_id,
                m.matcap_id(),
                m.metallic_roughness_texture_id,
                m.emissive_texture_id,
                m.selected_sampler(),
                prev_key,
                false,
                None,
            );
            if let Some((bg, key)) = built {
                entry.bind_group = Some(bg);
                entry.cache_key = key;
            }
        }
    }

    /// Rebuild or drop the cached batch render bundle for this frame.
    ///
    /// Eligible frames draw the batch list directly (no GPU-driven submission,
    /// no wireframe mode) with no registered deformers (their bind groups
    /// change under the recording) and enough opaque draws to be worth the
    /// recording. The bundle stores the opaque lean and fat batch draws in
    /// batch order; plugin batches and blended batches draw immediately after
    /// it. Per-instance transforms and colours flow through the instance
    /// buffer the recorded bind groups reference, so camera and object motion
    /// do not re-record; a batch-list rebuild, an instance-buffer
    /// reallocation or a rebuilt fat bind group does.
    pub(super) fn update_per_object_bundle(
        &mut self,
        device: &crate::gpu::Device,
        frame: &FrameData,
    ) {
        /// Below this many opaque draws the recording cost is not worth
        /// caching against.
        const MIN_BUNDLE_DRAWS: usize = 64;
        self.last_stats.per_object_bundle_cached = false;
        // Measurement kill-switch (`dev-knobs` feature) so the bundle can be
        // A/B'd in one binary.
        static DISABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *DISABLE.get_or_init(|| dev_knob!("VIEWPORT_DISABLE_PER_OBJECT_BUNDLE").is_some()) {
            self.per_object_bundle = None;
            return;
        }

        // Post-processing selects the HDR path; the bundle records against
        // that pass's formats and pipelines instead of being disabled. The
        // HDR pipelines build on the first HDR frame, so frame one records
        // nothing and the bundle starts on frame two.
        let hdr = frame.effects.display.is_hdr();
        let clipping_active = DeviceResources::clipping_active(frame);
        // GPU-driven submission draws the culled batches from cull-written
        // indirect args, so under it the bundle records the rest only.
        let fat_only =
            hdr && self.instancing.gpu_culling_enabled && self.resources.cull.hdr.is_some();
        let plan = 'plan: {
            if frame.viewport.wireframe_mode
                || self.instancing.batches.is_empty()
                || !self.resources.deform.meshes.is_empty()
            {
                break 'plan None;
            }
            if hdr && self.resources.scene.hdr.is_none() {
                break 'plan None;
            }
            let opaque = self
                .instancing
                .batches
                .iter()
                .filter(|b| {
                    !b.is_transparent
                        && b.shading_plugin.is_none()
                        && (!fat_only || !b.draws_culled())
                        && !b.fat.as_ref().is_some_and(|f| {
                            matches!(f.draw, crate::renderer::types::DrawKind::NormalLines)
                        })
                })
                .count();
            if opaque < MIN_BUNDLE_DRAWS {
                break 'plan None;
            }
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            hdr.hash(&mut h);
            fat_only.hash(&mut h);
            clipping_active.hash(&mut h);
            frame.camera.viewport_index.hash(&mut h);
            self.instancing.batches_gen.hash(&mut h);
            self.instancing.instance_gen.hash(&mut h);
            // Fat bind keys move with a mesh's override or lightmap
            // generation without a batch rebuild.
            for b in &self.instancing.batches {
                if let Some(f) = &b.fat {
                    f.bind_key.hash(&mut h);
                }
            }
            self.resources.force_po_discard.hash(&mut h);
            self.resources.mesh_universal.hash(&mut h);
            Some(h.finish())
        };

        let Some(key) = plan else {
            self.per_object_bundle = None;
            return;
        };

        // Churn gate. A single isolated change re-records immediately (the
        // measured cost of one re-record is a wash against immediate draws),
        // but a set that changes twice in short succession backs the bundle
        // off entirely: record + replay encodes every draw twice per frame,
        // and dropping a just-recorded bundle every frame leaks it in
        // wgpu 27 (gfx-rs/wgpu#8656). The gate re-arms once the set has
        // been stable for a stretch.
        const SUPPRESS_WINDOW: u32 = 8;
        const REARM_STABLE: u32 = 30;
        // A rebuilt bind group means the recorded one is stale even though
        // the key still matches, so it counts as a change too.
        let dirty = self.last_stats.per_object_bind_groups_built > 0;
        if self.per_object_bundle_gate.last_fat_only != Some(fat_only) {
            self.per_object_bundle_gate.last_fat_only = Some(fat_only);
            self.per_object_bundle_gate.frames_since_change = u32::MAX;
        }
        let changed = dirty || self.per_object_bundle_gate.last_key != Some(key);
        self.per_object_bundle_gate.last_key = Some(key);
        if changed {
            if self.per_object_bundle_gate.frames_since_change < SUPPRESS_WINDOW {
                self.per_object_bundle_gate.suppressed = true;
            }
            self.per_object_bundle_gate.frames_since_change = 0;
        } else {
            self.per_object_bundle_gate.frames_since_change = self
                .per_object_bundle_gate
                .frames_since_change
                .saturating_add(1);
            if self.per_object_bundle_gate.suppressed
                && self.per_object_bundle_gate.frames_since_change >= REARM_STABLE
            {
                self.per_object_bundle_gate.suppressed = false;
            }
        }
        // Diagnostic kill-switch for the churn gate (`dev-knobs` feature): with
        // it set, sustained churn re-records the bundle every frame instead of
        // backing off. Exists for leak retests on wgpu versions that ship the
        // render-bundle drop fix (gfx-rs/wgpu#8661); leave it unset
        // otherwise, since per-frame re-record costs more than it saves.
        static GATE_DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let gate_disabled = *GATE_DISABLED
            .get_or_init(|| dev_knob!("VIEWPORT_DISABLE_BUNDLE_CHURN_GATE").is_some());
        if self.per_object_bundle_gate.suppressed && !gate_disabled {
            self.per_object_bundle = None;
            return;
        }

        let camera_bg = self
            .viewport_camera_bind_group(frame.camera.viewport_index)
            .clone();
        let reusable = !dirty
            && self
                .per_object_bundle
                .as_ref()
                .is_some_and(|pb| pb.key == key && pb.camera_bg == camera_bg);
        if !reusable {
            match self.record_batch_bundle(device, key, camera_bg, hdr, fat_only, clipping_active) {
                Some(bundle) => self.per_object_bundle = Some(bundle),
                // A pipeline the list needs is still compiling: draw
                // immediately and try again next frame.
                None => {
                    self.per_object_bundle = None;
                    return;
                }
            }
        }
        self.last_stats.per_object_bundle_cached = true;
    }

    /// Record the opaque batch draws into a render bundle, as the direct
    /// opaque loop of the colour pass draws them. `None` when a pipeline is
    /// still compiling.
    fn record_batch_bundle(
        &self,
        device: &crate::gpu::Device,
        key: u64,
        camera_bg: crate::gpu::BindGroup,
        hdr: bool,
        fat_only: bool,
        clipping_active: bool,
    ) -> Option<crate::renderer::per_object_state::PerObjectBundle> {
        use crate::renderer::types::{FatBound, DrawKind, FatPass, MeshDrawEncoder, draw_fat_batch};
        let resources = &self.resources;
        // The HDR scene pass renders Rgba16Float at sample count 1 with a
        // writable stencil aspect; the LDR passes render the surface format
        // at the configured sample count with the stencil attached read-only
        // (`stencil_ops: None`). A read-only stencil declaration is valid in
        // both (the mesh pipelines never touch stencil).
        let (colour_format, sample_count) = if hdr {
            (crate::gpu::TextureFormat::Rgba16Float, 1)
        } else {
            (resources.target_format, resources.sample_count)
        };
        let mut enc = crate::resources::builders::render_bundle_encoder(
            device,
            "per_object_bundle",
            &[Some(colour_format)],
            Some(crate::gpu::RenderBundleDepthStencil {
                format: crate::resources::SCENE_DEPTH_FORMAT,
                depth_read_only: false,
                stencil_read_only: true,
            }),
            sample_count,
        );
        MeshDrawEncoder::set_bind_group(&mut enc, 0, &camera_bg, &[]);
        bind_deform_group!(enc, resources, &resources.deform.dummy_bind_group);

        // Every opaque batch the pass draws directly.
        let ldr_family = resources.scene.ldr_family();
        let hdr_family = resources.scene.hdr_family();
        let ldr_bindless_family = resources.scene.ldr_bindless_family();
        let hdr_bindless_family = resources.scene.hdr_bindless_family();
        let (family, bindless_family): (
            &dyn crate::resources::scene_pipelines::MeshColourFamily,
            &dyn crate::resources::scene_pipelines::MeshColourFamily,
        ) = if hdr {
            (&hdr_family, &hdr_bindless_family)
        } else {
            (&ldr_family, &ldr_bindless_family)
        };
        let fat_pass = FatPass::Colour {
            family,
            bindless_family,
            hdr,
            clipping_active,
        };
        let mut bound = FatBound::default();
        for batch in &self.instancing.batches {
            let Some(fat) = batch.fat.as_ref() else {
                continue;
            };
            if batch.is_transparent
                || batch.shading_plugin.is_some()
                || matches!(fat.draw, DrawKind::NormalLines)
                || (fat_only && batch.draws_culled())
            {
                continue;
            }
            if !draw_fat_batch(
                resources,
                &self.mesh_uniforms.material_bind_groups,
                &mut enc,
                batch,
                &fat_pass,
                &mut bound,
            ) || bound.stand_in
            {
                return None;
            }
        }

        let bundle = enc.finish(&crate::gpu::RenderBundleDescriptor {
            label: Some("per_object_bundle"),
        });
        Some(crate::renderer::per_object_state::PerObjectBundle {
            bundle,
            key,
            hdr,
            fat_only,
            camera_bg,
        })
    }
}
