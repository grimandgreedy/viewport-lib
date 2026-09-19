//! An item type that does everything an item type does, written against
//! nothing but `viewport-lib`'s public API.
//!
//! The library's own item types live inside `viewport-lib`, so nothing stops
//! one of them reaching a `pub(crate)` route by accident. That has happened: an
//! upload path was written that needed the plugin, the job runner and the
//! shared content arenas at once, and the only way to get all three was a
//! constructor no crate outside the library could call. Reading the code did
//! not catch it; this crate depends on `viewport-lib` the way anyone else's
//! does, so it would have.
//!
//! The other fixtures here each cover one part of that. [`ConformanceItemTypePlugin`]
//! is the whole of it in one type, so the parts are checked together rather
//! than one at a time. It is not a useful item type: it draws a textured quad
//! per stored entry and nothing more. What it does is everything:
//!
//! - holds a store of its own, filled through
//!   [`ViewportRenderer::item_type_plugin_host`](viewport_lib::ViewportRenderer::item_type_plugin_host)
//!   and read back through
//!   [`item_type_plugin`](viewport_lib::ViewportRenderer::item_type_plugin)
//! - builds its buffers off the frame thread on the library's job runner
//! - draws in the scene pass, the pick pass, the shadow pass and the
//!   outline-mask pass, from pipelines built by the published builders
//! - answers a CPU pick, and returns a bounds wireframe for wireframe mode and
//!   for selection
//! - bakes a host texture into a bind group it keeps across frames, and
//!   revalidates it with a [`ResourceGate`] when the host frees or replaces
//!   that texture
//! - reports what it holds to the renderer's working-set figure
//!
//! Every one of those is a public-API claim. If any of them stops being
//! reachable from outside the library, this file stops compiling.
//!
//! It is broader than a fixture usually is, and deliberately so. The others
//! here are each minimal against one hook, which is right for proving that hook
//! fires; what none of them can show is whether the hooks are usable *together*
//! by one type, which is the shape a real item type has and the shape the gaps
//! turn up in. Breadth across the seam, not features: there is still nothing
//! believable about a flat quad.

use std::any::Any;

use viewport_lib::plugin_api::shared_wgsl::{
    SHARED_BINDINGS_WGSL, SHARED_MASK_WGSL, SHARED_PICK_WGSL, SHARED_SHADOW_BINDINGS_WGSL,
};
use viewport_lib::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickContext,
    PickPassContext, PickRay, PluginItemCollection, ShadowCastContext,
};
use viewport_lib::resources::{
    DeviceResources, JobId, Jobs, PluginPipelineOpts, ResourceGate, Revalidate, TextureId,
    UploadStatus,
};
use viewport_lib::{ItemSettings, PickHit, PickId, PolylineItem, wgpu};

// Two public enums are called `TextureSlot` and neither accepts the other.
// `viewport_lib::TextureSlot` names a slot in the material's own layout and is
// what `fallback_texture_view` wants; `viewport_lib::resources::TextureSlot`
// names a slot across every item type and is what `check_texture_slot` wants.
// Aliased rather than imported so each call site says which it means.
use viewport_lib::TextureSlot as MaterialSlot;
use viewport_lib::resources::TextureSlot as ReportedSlot;

/// Half-extent of the quad each stored entry draws, in world units.
const HALF: f32 = 0.9;

/// Byte offset of the pick id inside [`QuadUniform`]: two `vec4`s in.
const PICK_ID_OFFSET: u64 = 32;

/// Handle to one entry in the plugin's own store.
///
/// A bare index plus a generation. The generation is the part worth copying: a
/// bare index aliases after a slot is reused, so a handle the host is still
/// holding would resolve to somebody else's content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuadId {
    index: usize,
    generation: u32,
}

/// What one stored entry needs on the GPU.
struct Entry {
    generation: u32,
    /// CPU mirror of the uniform's centre, so `pick` and `wireframe_polylines`
    /// can answer without reading the buffer back.
    centre: glam::Vec3,
    /// CPU mirror of the pick id last written into the uniform.
    pick_id: PickId,
    /// Centre, colour and pick id, written on upload and rewritten when the
    /// host changes the item's pick id.
    uniform: wgpu::Buffer,
    /// The host texture this entry samples, or `None` once it has gone away.
    /// Nulled rather than remembered: a dead id must not be looked up again,
    /// and a generational id freed and reused would otherwise resolve to
    /// somebody else's texture.
    texture: Option<TextureId>,
    /// Group 1 for every pass: the uniform, a texture view and a sampler. This
    /// is what pins a freed texture if nothing revalidates it.
    bind_group: wgpu::BindGroup,
    bytes: u64,
}

/// The per-frame items of this type, submitted with
/// [`SceneFrame::submit_plugin_items`](viewport_lib::renderer::SceneFrame::submit_plugin_items).
///
/// Each item names a stored entry and carries the shared
/// [`ItemSettings`]: the plugin reads `hidden`, `selected`, `wireframe` and
/// `pick_id` out of them the way the built-in types do.
#[derive(Default)]
pub struct ConformanceItems {
    settings: Vec<ItemSettings>,
    quads: Vec<QuadId>,
}

impl ConformanceItems {
    /// Submit one stored entry with default settings.
    pub fn new(quad: QuadId) -> Self {
        let mut items = Self::default();
        items.push(quad, ItemSettings::default());
        items
    }

    /// Submit one stored entry with settings of your own.
    pub fn push(&mut self, quad: QuadId, settings: ItemSettings) {
        self.quads.push(quad);
        self.settings.push(settings);
    }
}

impl PluginItemCollection for ConformanceItems {
    fn len(&self) -> usize {
        self.quads.len()
    }

    fn item_settings(&self, index: usize) -> &ItemSettings {
        &self.settings[index]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The bytes each entry's uniform holds: centre, colour and pick id, one
/// `vec4` each so the layout needs no padding reasoning. Packed by hand rather
/// than through a `Pod` derive, to keep this fixture's dependencies to the
/// library itself.
const UNIFORM_BYTES: usize = 48;

fn pack_uniform(centre: glam::Vec3, colour: [f32; 3], pick_id: PickId) -> [u8; UNIFORM_BYTES] {
    let mut out = [0u8; UNIFORM_BYTES];
    let floats = [
        centre.x, centre.y, centre.z, 0.0, colour[0], colour[1], colour[2], 1.0,
    ];
    for (i, f) in floats.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&f.to_ne_bytes());
    }
    out[PICK_ID_OFFSET as usize..PICK_ID_OFFSET as usize + 4]
        .copy_from_slice(&(pick_id.0 as u32).to_ne_bytes());
    out
}

/// What an off-thread upload builds before the main thread can insert it.
struct BuiltQuad {
    uniform: wgpu::Buffer,
    centre: glam::Vec3,
    texture: Option<TextureId>,
    bytes: u64,
}

/// The item type. See the module docs for what it is for.
pub struct ConformanceItemTypePlugin {
    opaque: wgpu::RenderPipeline,
    pick: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    mask: wgpu::RenderPipeline,
    entry_bgl: wgpu::BindGroupLayout,
    store: Vec<Option<Entry>>,
    next_generation: u32,
    gate: ResourceGate,
    rebinds: u32,
}

impl ConformanceItemTypePlugin {
    /// The name this type registers under.
    pub const TYPE_NAME: &'static str = "conformance.quad";

    /// Build the pipelines and return the plugin, ready to register with
    /// [`with_item_type_plugin`](viewport_lib::ViewportRenderer::with_item_type_plugin).
    ///
    /// The pipeline builders live on [`DeviceResources`], which
    /// [`ItemTypePlugin::init_gpu`] is not handed, so a plugin that uses them is
    /// constructed by the host, which has the renderer. That is the same shape
    /// the library's own renderer-side handles use.
    pub fn new(resources: &DeviceResources, device: &wgpu::Device) -> Self {
        let entry_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conformance_entry_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let extra = [&entry_bgl];

        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conformance_scene_shader"),
            source: wgpu::ShaderSource::Wgsl(scene_wgsl().into()),
        });
        let mut opaque_opts = PluginPipelineOpts::new(
            Some("conformance_opaque"),
            &scene_shader,
            "vs_scene",
            "fs_scene",
            &[],
        );
        opaque_opts.extra_bind_group_layouts = &extra;
        // A flat quad has no inside, so neither winding is a back face.
        opaque_opts.primitive.cull_mode = None;
        let opaque = resources.build_opaque_pipeline(device, &opaque_opts);

        let mut pick_opts = PluginPipelineOpts::new(
            Some("conformance_pick"),
            &scene_shader,
            "vs_pick",
            "viewport_pick_fs",
            &[],
        );
        pick_opts.extra_bind_group_layouts = &extra;
        pick_opts.primitive.cull_mode = None;
        let pick = resources.build_pick_pipeline(device, &pick_opts);

        let mut mask_opts = PluginPipelineOpts::new(
            Some("conformance_mask"),
            &scene_shader,
            "vs_scene",
            "viewport_mask_fs",
            &[],
        );
        mask_opts.extra_bind_group_layouts = &extra;
        mask_opts.primitive.cull_mode = None;
        let mask = resources.build_mask_pipeline(device, &mask_opts);

        // The shadow pass binds a different group 0, so its stage lives in its
        // own module with the shadow declarations at the top.
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conformance_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(shadow_wgsl().into()),
        });
        let mut shadow_opts = PluginPipelineOpts::new(
            Some("conformance_shadow"),
            &shadow_shader,
            "vs_shadow",
            "",
            &[],
        );
        shadow_opts.extra_bind_group_layouts = &extra;
        shadow_opts.primitive.cull_mode = None;
        let shadow = resources.build_shadow_pipeline(device, &shadow_opts);

        Self {
            opaque,
            pick,
            shadow,
            mask,
            entry_bgl,
            store: Vec::new(),
            next_generation: 1,
            gate: ResourceGate::default(),
            rebinds: 0,
        }
    }

    /// Upload one quad and return its handle.
    ///
    /// `texture` is validated against the shared texture store before it is
    /// baked in, which an item type binding a host's texture has to do: the
    /// store belongs to the library, not to the plugin.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &DeviceResources,
        centre: glam::Vec3,
        colour: [f32; 3],
        texture: Option<TextureId>,
    ) -> QuadId {
        let built = build_quad(device, queue, resources, centre, colour, texture);
        self.insert(built, device, resources)
    }

    /// Start an off-thread upload. Poll the returned job with
    /// [`upload_status`](viewport_lib::ViewportRenderer::upload_status) and take
    /// the handle from [`take_upload`](Self::take_upload) once it reports
    /// `Ready`.
    ///
    /// The texture id is resolved here, on the calling thread, and only the
    /// validated id crosses to the worker. Anything borrowed from
    /// `DeviceResources` has to be resolved before the closure is built: the
    /// borrow cannot outlive this call, and a worker that held one would be
    /// reading a store the main thread is free to mutate.
    pub fn begin_upload(
        &self,
        jobs: &Jobs<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &DeviceResources,
        centre: glam::Vec3,
        colour: [f32; 3],
        texture: Option<TextureId>,
    ) -> JobId {
        let texture = texture.filter(|id| resources.has_texture(*id));
        let device = device.clone();
        let queue = queue.clone();
        jobs.try_submit_cpu(move |progress| {
            progress.set(0.5);
            Ok(PendingQuad {
                uniform: uniform_buffer(&device, &queue, centre, colour, PickId::NONE),
                centre,
                texture,
            })
        })
    }

    /// Take a finished off-thread upload into the store.
    ///
    /// The handle is minted here rather than in `begin_upload` because this is
    /// the call holding `&mut self` to insert with.
    pub fn take_upload(
        &mut self,
        jobs: &Jobs<'_>,
        device: &wgpu::Device,
        resources: &DeviceResources,
        id: JobId,
    ) -> Option<QuadId> {
        if !matches!(jobs.status(id), UploadStatus::Ready) {
            return None;
        }
        let pending = jobs.take::<PendingQuad>(id)?;
        let bytes = pending.uniform.size();
        Some(self.insert(
            BuiltQuad {
                uniform: pending.uniform,
                centre: pending.centre,
                texture: pending.texture,
                bytes,
            },
            device,
            resources,
        ))
    }

    /// Whether a handle still resolves. A handle to a freed and reused slot
    /// does not, which is what the generation is for.
    pub fn contains(&self, id: QuadId) -> bool {
        self.entry(id).is_some()
    }

    /// The texture a stored entry samples, or `None` if it never named one or
    /// the host has since freed it.
    pub fn texture_of(&self, id: QuadId) -> Option<TextureId> {
        self.entry(id)?.texture
    }

    /// How many bind groups this plugin has had to rebuild because a resource
    /// it had baked in went away. Observable so a test can tell a revalidation
    /// that ran from one that happened not to be needed.
    pub fn rebind_count(&self) -> u32 {
        self.rebinds
    }

    /// Release a stored entry. The slot is reused, and the generation moves so
    /// the old handle stops resolving.
    pub fn free(&mut self, id: QuadId) -> bool {
        match self.store.get_mut(id.index) {
            Some(slot) if slot.as_ref().is_some_and(|e| e.generation == id.generation) => {
                *slot = None;
                true
            }
            _ => false,
        }
    }

    fn entry(&self, id: QuadId) -> Option<&Entry> {
        self.store
            .get(id.index)?
            .as_ref()
            .filter(|e| e.generation == id.generation)
    }

    fn insert(
        &mut self,
        built: BuiltQuad,
        device: &wgpu::Device,
        resources: &DeviceResources,
    ) -> QuadId {
        let generation = self.next_generation;
        self.next_generation += 1;
        let bind_group = self.bind(&built.uniform, built.texture, device, resources);
        let entry = Entry {
            generation,
            centre: built.centre,
            pick_id: PickId::NONE,
            uniform: built.uniform,
            texture: built.texture,
            bind_group,
            bytes: built.bytes,
        };
        let index = match self.store.iter().position(|slot| slot.is_none()) {
            Some(index) => {
                self.store[index] = Some(entry);
                index
            }
            None => {
                self.store.push(Some(entry));
                self.store.len() - 1
            }
        };
        QuadId { index, generation }
    }

    /// Build group 1 for an entry. A texture that no longer resolves falls back
    /// to the library's neutral view, so the layout is honoured either way.
    fn bind(
        &self,
        uniform: &wgpu::Buffer,
        texture: Option<TextureId>,
        device: &wgpu::Device,
        resources: &DeviceResources,
    ) -> wgpu::BindGroup {
        let view = texture
            .and_then(|id| resources.texture_view(id))
            .unwrap_or_else(|| resources.fallback_texture_view(MaterialSlot::Albedo));
        resources.check_texture_slot(texture, ReportedSlot::MaterialAlbedo);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conformance_entry_bg"),
            layout: &self.entry_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(resources.material_sampler()),
                },
            ],
        })
    }

    /// Poll the gate and rebuild whatever it says is stale.
    ///
    /// Rebind rather than discard: the entry belongs to the host, which holds a
    /// handle to it and did not ask for it to go away. A freed texture is
    /// replaced by the fallback view and the dead id is forgotten, so the entry
    /// keeps drawing and never looks the id up again.
    fn revalidate(&mut self, device: &wgpu::Device, resources: &DeviceResources) {
        let verdict = self.gate.poll(resources);
        if verdict == Revalidate::Valid {
            return;
        }
        for index in 0..self.store.len() {
            let Some(entry) = self.store[index].as_ref() else {
                continue;
            };
            let dead = entry.texture.is_some_and(|id| !resources.has_texture(id));
            if verdict == Revalidate::CheckEach && !dead {
                continue;
            }
            let texture = if dead { None } else { entry.texture };
            let bind_group = self.bind(&entry.uniform, texture, device, resources);
            let entry = self.store[index].as_mut().expect("checked just above");
            entry.texture = texture;
            entry.bind_group = bind_group;
            self.rebinds += 1;
        }
    }

    /// Draw every submitted item that resolves and is not hidden.
    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        items: &dyn PluginItemCollection,
        selected_only: bool,
    ) {
        let Some(items) = items.as_any().downcast_ref::<ConformanceItems>() else {
            return;
        };
        pass.set_pipeline(pipeline);
        for (index, quad) in items.quads.iter().enumerate() {
            let settings = &items.settings[index];
            if settings.hidden || (selected_only && !settings.selected) {
                continue;
            }
            let Some(entry) = self.entry(*quad) else {
                continue;
            };
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

/// The value an off-thread upload hands back. Separate from [`BuiltQuad`]
/// because the byte figure is taken on the main thread, where the store is.
struct PendingQuad {
    uniform: wgpu::Buffer,
    centre: glam::Vec3,
    texture: Option<TextureId>,
}

impl ItemTypePlugin for ConformanceItemTypePlugin {
    fn type_name(&self) -> &'static str {
        Self::TYPE_NAME
    }

    fn resident_bytes(&self) -> u64 {
        self.store.iter().flatten().map(|e| e.bytes).sum()
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<wgpu::CommandBuffer> {
        // Before anything reads a stored bind group: a texture the host freed
        // or replaced since the last frame has to be dealt with first.
        self.revalidate(device, ctx.resources);

        // The pick pass reads the id out of the entry uniform, and the host is
        // free to change an item's pick id between frames.
        if let Some(items) = items.as_any().downcast_ref::<ConformanceItems>() {
            for (index, quad) in items.quads.iter().enumerate() {
                let pick_id = items.settings[index].pick_id;
                let Some(slot) = self.store.get_mut(quad.index) else {
                    continue;
                };
                let Some(entry) = slot.as_mut().filter(|e| e.generation == quad.generation) else {
                    continue;
                };
                if entry.pick_id == pick_id {
                    continue;
                }
                entry.pick_id = pick_id;
                queue.write_buffer(
                    &entry.uniform,
                    PICK_ID_OFFSET,
                    &(pick_id.0 as u32).to_ne_bytes(),
                );
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.draw(pass, &self.opaque, items, false);
    }

    fn cast_shadow_pass(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        _ctx: &ShadowCastContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.draw(pass, &self.shadow, items, false);
    }

    fn render_pick(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        _ctx: &PickPassContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.draw(pass, &self.pick, items, false);
    }

    fn outline_mask(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        items: &dyn PluginItemCollection,
    ) {
        self.draw(pass, &self.mask, items, true);
    }

    fn wireframe_polylines(
        &self,
        items: &dyn PluginItemCollection,
        ctx: &ItemFrameContext<'_>,
    ) -> Vec<PolylineItem> {
        let Some(items) = items.as_any().downcast_ref::<ConformanceItems>() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (index, quad) in items.quads.iter().enumerate() {
            let settings = &items.settings[index];
            if settings.hidden {
                continue;
            }
            // Selection is the brighter of the two, and takes precedence: a
            // selected item in wireframe mode shows it is selected.
            let colour = if settings.selected {
                [1.0, 0.9, 0.2, 1.0]
            } else if settings.wireframe || ctx.wireframe_mode {
                [0.7, 0.7, 0.8, 1.0]
            } else {
                continue;
            };
            let Some(entry) = self.entry(*quad) else {
                continue;
            };
            let centre = centre_of(entry);
            let mut line = viewport_lib::aabb_wireframe_polyline(&bounds_at(centre), colour);
            line.settings.wireframe = true;
            out.push(line);
        }
        out
    }

    fn pick(&self, ray: &PickRay, _ctx: &PickContext<'_>) -> Option<(f32, PickHit)> {
        // Answered from the store, not from the frame: pick runs out of band
        // with prepare, so the collection is not available here.
        let mut best: Option<(f32, PickHit)> = None;
        for entry in self.store.iter().flatten() {
            let centre = centre_of(entry);
            let Some(toi) = ray_quad_toi(ray, centre) else {
                continue;
            };
            let id = pick_id_of(entry);
            if id == PickId::NONE {
                continue;
            }
            if best.as_ref().is_none_or(|(t, _)| toi < *t) {
                let hit =
                    PickHit::object_hit(id.0, ray.origin + ray.direction * toi, glam::Vec3::Z);
                best = Some((toi, hit));
            }
        }
        best
    }
}

/// The world-space centre an entry was uploaded at, read back from its uniform
/// mirror. Kept beside the buffer rather than read from the GPU.
fn centre_of(entry: &Entry) -> glam::Vec3 {
    entry.centre
}

/// The pick id currently written into an entry's uniform.
fn pick_id_of(entry: &Entry) -> PickId {
    entry.pick_id
}

/// The quad's bounds, for the wireframe hook.
fn bounds_at(centre: glam::Vec3) -> viewport_lib::Aabb {
    viewport_lib::Aabb {
        min: centre - glam::Vec3::new(HALF, HALF, 0.0),
        max: centre + glam::Vec3::new(HALF, HALF, 0.0),
    }
}

/// Ray against the quad's plane, then a bounds test in it.
fn ray_quad_toi(ray: &PickRay, centre: glam::Vec3) -> Option<f32> {
    let normal = glam::Vec3::Z;
    let denom = ray.direction.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let toi = (centre - ray.origin).dot(normal) / denom;
    if toi < 0.0 {
        return None;
    }
    let hit = ray.origin + ray.direction * toi;
    let d = hit - centre;
    (d.x.abs() <= HALF && d.y.abs() <= HALF).then_some(toi)
}

fn uniform_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    centre: glam::Vec3,
    colour: [f32; 3],
    pick_id: PickId,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("conformance_quad_uniform"),
        size: UNIFORM_BYTES as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, &pack_uniform(centre, colour, pick_id));
    buffer
}

fn build_quad(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &DeviceResources,
    centre: glam::Vec3,
    colour: [f32; 3],
    texture: Option<TextureId>,
) -> BuiltQuad {
    let texture = texture.filter(|id| resources.has_texture(*id));
    let uniform = uniform_buffer(device, queue, centre, colour, PickId::NONE);
    let bytes = uniform.size();
    BuiltQuad {
        uniform,
        centre,
        texture,
        bytes,
    }
}

/// The scene, pick and mask stages. `SHARED_BINDINGS_WGSL` supplies group 0, so
/// nothing here re-declares it.
fn scene_wgsl() -> String {
    format!(
        "{SHARED_BINDINGS_WGSL}
{SHARED_PICK_WGSL}
{SHARED_MASK_WGSL}

struct QuadUniform {{
    centre: vec4<f32>,
    colour: vec4<f32>,
    pick_id: vec4<u32>,
}};
@group(1) @binding(0) var<uniform> quad: QuadUniform;
@group(1) @binding(1) var quad_tex: texture_2d<f32>;
@group(1) @binding(2) var quad_samp: sampler;

// Two triangles in the world XY plane, generated from the vertex index so the
// plugin needs no vertex buffer. Z-up, so the quad stands upright and faces +Z,
// which is what the default top-down view looks at.
fn quad_corner(vertex_index: u32) -> vec2<f32> {{
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    return corners[vertex_index];
}}

fn quad_world(vertex_index: u32) -> vec3<f32> {{
    let c = quad_corner(vertex_index) * {HALF};
    return quad.centre.xyz + vec3<f32>(c.x, c.y, 0.0);
}}

struct SceneVsOut {{
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}};

@vertex
fn vs_scene(@builtin(vertex_index) vi: u32) -> SceneVsOut {{
    var out: SceneVsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(quad_world(vi), 1.0);
    out.uv = quad_corner(vi) * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}}

@fragment
fn fs_scene(in: SceneVsOut) -> @location(0) vec4<f32> {{
    let tex = textureSample(quad_tex, quad_samp, in.uv);
    return vec4<f32>(quad.colour.rgb * tex.rgb, 1.0);
}}

struct PickVsOut {{
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) pick_id: u32,
}};

@vertex
fn vs_pick(@builtin(vertex_index) vi: u32) -> PickVsOut {{
    var out: PickVsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(quad_world(vi), 1.0);
    out.pick_id = quad.pick_id.x;
    return out;
}}
"
    )
}

/// The shadow-cast stage, in its own module: the shadow pass binds a different
/// group 0, so this cannot share a module with the rest. Depth-only, so there
/// is no fragment stage.
fn shadow_wgsl() -> String {
    format!(
        "{SHARED_SHADOW_BINDINGS_WGSL}

struct QuadUniform {{
    centre: vec4<f32>,
    colour: vec4<f32>,
    pick_id: vec4<u32>,
}};
@group(1) @binding(0) var<uniform> quad: QuadUniform;
@group(1) @binding(1) var quad_tex: texture_2d<f32>;
@group(1) @binding(2) var quad_samp: sampler;

@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi] * {HALF};
    let world = quad.centre.xyz + vec3<f32>(c.x, c.y, 0.0);
    return shadow_camera.light_view_proj * vec4<f32>(world, 1.0);
}}
"
    )
}
