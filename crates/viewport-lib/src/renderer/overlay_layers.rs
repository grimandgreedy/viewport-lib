//! Overlay layers: which items draw into a scratch target rather than the
//! frame, the scratch targets, and the passes that fill, blur and read them.
//!
//! A layer that has nothing to do as a whole folds away before anything is
//! drawn: a pass-through layer leaves its members alone, and a layer with one
//! immediate member pushes its opacity, blend and colour matrix onto that
//! member. Only the rest (and every item whose blend reads the destination)
//! draws into a scratch target.
//!
//! A scratch target is the viewport's size, so members draw at their screen
//! positions with no change to their shaders and the composite reads them back
//! texel for texel. Layers whose padded bounds are disjoint share one; a layer
//! that blurs its content or casts a shadow has one to itself. The targets are
//! pooled per viewport and kept across frames.

use crate::renderer::overlay_draw_order::{Bounds, NO_LAYER, intersect, overlaps, union};
use crate::renderer::types::{
    OverlayBlend, OverlayClip, OverlayFrame, OverlayLayerStyle, OverlayStyle,
};
use crate::resources::overlay::overlay_layer::{LayerPipelineKey, OverlayLayerInstance};

/// How a layer draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LayerMode {
    /// Its members draw into the frame unchanged.
    Passthrough,
    /// Its one member draws into the frame with the layer's opacity, blend and
    /// colour matrix applied to it.
    Push,
    /// Its members draw into a scratch target that is composited as one.
    Target,
}

/// A layer as the frame resolved it.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedLayer {
    pub style: OverlayLayerStyle,
    pub z_order: i32,
    pub clip: OverlayClip,
    pub mode: LayerMode,
}

/// The item lists a layer can name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Family {
    Shape,
    Label,
    GlyphRun,
    Polyline,
    Mesh,
    Retained,
}

/// Every layer of a frame, and the layer each item belongs to.
#[derive(Default)]
pub(crate) struct LayerPlan {
    pub layers: Vec<ResolvedLayer>,
    shapes: Vec<u32>,
    labels: Vec<u32>,
    glyph_runs: Vec<u32>,
    polylines: Vec<u32>,
    meshes: Vec<u32>,
    retained: Vec<u32>,
}

impl LayerPlan {
    fn list(&self, family: Family) -> &Vec<u32> {
        match family {
            Family::Shape => &self.shapes,
            Family::Label => &self.labels,
            Family::GlyphRun => &self.glyph_runs,
            Family::Polyline => &self.polylines,
            Family::Mesh => &self.meshes,
            Family::Retained => &self.retained,
        }
    }

    fn list_mut(&mut self, family: Family) -> &mut Vec<u32> {
        match family {
            Family::Shape => &mut self.shapes,
            Family::Label => &mut self.labels,
            Family::GlyphRun => &mut self.glyph_runs,
            Family::Polyline => &mut self.polylines,
            Family::Mesh => &mut self.meshes,
            Family::Retained => &mut self.retained,
        }
    }

    /// The layer item `index` of `family` belongs to, or [`NO_LAYER`].
    pub(crate) fn layer_of(&self, family: Family, index: usize) -> u32 {
        self.list(family).get(index).copied().unwrap_or(NO_LAYER)
    }

    /// The layer the item's draws are recorded into: its layer when that
    /// draws through a scratch target, [`NO_LAYER`] (the frame) otherwise.
    pub(crate) fn record_layer(&self, family: Family, index: usize) -> u32 {
        let l = self.layer_of(family, index);
        match self.layers.get(l as usize) {
            Some(layer) if layer.mode == LayerMode::Target => l,
            _ => NO_LAYER,
        }
    }

    /// Whether any layer draws through a scratch target.
    pub(crate) fn has_targets(&self) -> bool {
        self.layers.iter().any(|l| l.mode == LayerMode::Target)
    }

    /// The style item `index` of `family` draws with, when its layer changes
    /// it: a pushed layer's opacity, blend and colour matrix, or, inside a
    /// scratch target, a blend that reads the destination drawn as
    /// source-over and no backdrop. `None` when the item's own style stands.
    pub(crate) fn adjusted_style(
        &self,
        family: Family,
        index: usize,
        style: &OverlayStyle,
    ) -> Option<OverlayStyle> {
        let l = self.layer_of(family, index);
        let layer = self.layers.get(l as usize)?;
        match layer.mode {
            LayerMode::Passthrough => None,
            LayerMode::Push => {
                let mut s = style.clone();
                s.opacity *= layer.style.opacity;
                if layer.style.blend != OverlayBlend::SourceOver {
                    s.blend = layer.style.blend;
                }
                if s.colour_matrix.is_none() {
                    s.colour_matrix = layer.style.colour_matrix;
                }
                Some(s)
            }
            LayerMode::Target => {
                if !style.blend.reads_destination() && !style.backdrop.is_active() {
                    return None;
                }
                let mut s = style.clone();
                if s.blend.reads_destination() {
                    s.blend = OverlayBlend::SourceOver;
                }
                s.backdrop = Default::default();
                Some(s)
            }
        }
    }

    /// Resolve `frame`'s layers. Cheap when the frame has none: the per-item
    /// lists stay empty and every lookup answers [`NO_LAYER`].
    pub(crate) fn build(&mut self, frame: &OverlayFrame) {
        self.layers.clear();
        for f in [
            Family::Shape,
            Family::Label,
            Family::GlyphRun,
            Family::Polyline,
            Family::Mesh,
            Family::Retained,
        ] {
            self.list_mut(f).clear();
        }
        let reads = |s: &OverlayStyle| s.blend.reads_destination();
        let any = !frame.layers.is_empty()
            || frame.retained.iter().any(|r| r.layer.is_some())
            || frame
                .shapes
                .iter()
                .any(|s| s.provides_mask.is_none() && reads(&s.style))
            || frame.labels.iter().any(|s| reads(&s.style))
            || frame.glyph_runs.iter().any(|s| reads(&s.style))
            || frame.polylines.iter().any(|s| reads(&s.style))
            || frame.meshes.iter().any(|s| reads(&s.style));
        if !any {
            return;
        }
        self.shapes.resize(frame.shapes.len(), NO_LAYER);
        self.labels.resize(frame.labels.len(), NO_LAYER);
        self.glyph_runs.resize(frame.glyph_runs.len(), NO_LAYER);
        self.polylines.resize(frame.polylines.len(), NO_LAYER);
        self.meshes.resize(frame.meshes.len(), NO_LAYER);
        self.retained.resize(frame.retained.len(), NO_LAYER);

        // Explicit layers: an item belongs to the first layer naming it.
        for layer in &frame.layers {
            let idx = self.layers.len() as u32;
            self.layers.push(ResolvedLayer {
                style: layer.style.clone(),
                z_order: layer.z_order,
                clip: layer.clip,
                mode: LayerMode::Target,
            });
            let m = &layer.members;
            for (family, range) in [
                (Family::Shape, m.shapes.clone()),
                (Family::Label, m.labels.clone()),
                (Family::GlyphRun, m.glyph_runs.clone()),
                (Family::Polyline, m.polylines.clone()),
                (Family::Mesh, m.meshes.clone()),
                (Family::Retained, m.retained.clone()),
            ] {
                let list = self.list_mut(family);
                let end = range.end.min(list.len());
                for slot in &mut list[range.start.min(end)..end] {
                    if *slot == NO_LAYER {
                        *slot = idx;
                    }
                }
            }
        }
        // A retained group with a layer style is a layer of its own.
        for (i, r) in frame.retained.iter().enumerate() {
            if let (Some(style), NO_LAYER) = (&r.layer, self.retained[i]) {
                self.retained[i] = self.layers.len() as u32;
                self.layers.push(ResolvedLayer {
                    style: style.clone(),
                    z_order: r.z_order,
                    clip: r.clip,
                    mode: LayerMode::Target,
                });
            }
        }
        // An item outside every layer whose blend reads the destination is a
        // one-item layer with that blend.
        let implicit = |plan: &mut Self, family: Family, i: usize, style: &OverlayStyle, z| {
            if reads(style) && plan.list(family)[i] == NO_LAYER {
                plan.list_mut(family)[i] = plan.layers.len() as u32;
                plan.layers.push(ResolvedLayer {
                    style: OverlayLayerStyle::default().with_blend(style.blend),
                    z_order: z,
                    clip: OverlayClip::default(),
                    mode: LayerMode::Target,
                });
            }
        };
        for (i, s) in frame.shapes.iter().enumerate() {
            if s.provides_mask.is_none() {
                implicit(self, Family::Shape, i, &s.style, s.z_order);
            }
        }
        for (i, s) in frame.labels.iter().enumerate() {
            implicit(self, Family::Label, i, &s.style, s.z_order);
        }
        for (i, s) in frame.glyph_runs.iter().enumerate() {
            implicit(self, Family::GlyphRun, i, &s.style, s.z_order);
        }
        for (i, s) in frame.polylines.iter().enumerate() {
            implicit(self, Family::Polyline, i, &s.style, s.z_order);
        }
        for (i, s) in frame.meshes.iter().enumerate() {
            implicit(self, Family::Mesh, i, &s.style, s.z_order);
        }

        // Count each layer's drawn members and remember a lone immediate one.
        let mut counts = vec![(0usize, None::<(Family, usize)>); self.layers.len()];
        let mut tally = |l: u32, member: Option<(Family, usize)>| {
            if let Some(c) = counts.get_mut(l as usize) {
                c.0 += 1;
                c.1 = member;
            }
        };
        for (i, s) in frame.shapes.iter().enumerate() {
            if s.provides_mask.is_none() {
                tally(self.shapes[i], Some((Family::Shape, i)));
            }
        }
        for (i, l) in self.labels.iter().enumerate() {
            tally(*l, Some((Family::Label, i)));
        }
        for (i, l) in self.glyph_runs.iter().enumerate() {
            tally(*l, Some((Family::GlyphRun, i)));
        }
        for (i, l) in self.polylines.iter().enumerate() {
            tally(*l, Some((Family::Polyline, i)));
        }
        for (i, l) in self.meshes.iter().enumerate() {
            tally(*l, Some((Family::Mesh, i)));
        }
        for l in &self.retained {
            tally(*l, None);
        }
        let style_of = |member: (Family, usize)| -> &OverlayStyle {
            match member {
                (Family::Shape, i) => &frame.shapes[i].style,
                (Family::Label, i) => &frame.labels[i].style,
                (Family::GlyphRun, i) => &frame.glyph_runs[i].style,
                (Family::Polyline, i) => &frame.polylines[i].style,
                (Family::Mesh, i) => &frame.meshes[i].style,
                (Family::Retained, _) => unreachable!("a retained member is not pushed"),
            }
        };
        for (layer, (count, member)) in self.layers.iter_mut().zip(counts) {
            let lone = if count == 1 { member } else { None };
            let member_style = lone.map(style_of);
            // A lone member that reads the destination lends the layer its
            // blend when the layer has none of its own.
            if let Some(ms) = member_style {
                if ms.blend.reads_destination() && layer.style.blend == OverlayBlend::SourceOver {
                    layer.style.blend = ms.blend;
                }
            }
            let style = &layer.style;
            layer.mode = if count == 0 || style.is_passthrough() {
                LayerMode::Passthrough
            } else if style.needs_target() {
                LayerMode::Target
            } else if member_style.is_some_and(|ms| {
                !ms.blend.reads_destination()
                    && (style.blend == OverlayBlend::SourceOver
                        || ms.blend == OverlayBlend::SourceOver)
                    && !(style.colour_matrix.is_some() && ms.colour_matrix.is_some())
            }) {
                LayerMode::Push
            } else {
                LayerMode::Target
            };
        }
    }
}

/// How far the backdrop blur reaches from a pixel, in logical pixels, for a
/// blur `radius`: six taps either side at half resolution.
pub(crate) fn blur_reach(radius: f32, ppp: f32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    6.0 * radius.max(2.0) / ppp.max(1e-3) + 2.0
}

/// The tap spread (half-resolution texels) of a Gaussian of standard deviation
/// `sigma` logical pixels: the kernel's standard deviation is three taps, a
/// half-resolution texel two device pixels.
fn gaussian_spread(sigma: f32, ppp: f32) -> f32 {
    sigma * ppp / 6.0
}

/// The standard deviation of a layer's content blur of `radius`, which is the
/// CSS `blur()` length.
fn content_sigma(radius: f32) -> f32 {
    radius
}

/// The standard deviation of a layer shadow of `blur`, which is the fade
/// width an item's shadow layer uses, as CSS `box-shadow` does.
fn shadow_sigma(blur: f32) -> f32 {
    blur * 0.5
}

/// How far a Gaussian of standard deviation `sigma` reaches: the kernel stops
/// at two, plus a pixel of bilinear footprint either side.
fn gaussian_reach(sigma: f32, ppp: f32) -> f32 {
    if sigma <= 0.0 {
        return 0.0;
    }
    2.0 * sigma + 2.0 / ppp.max(1e-3)
}

fn expand(b: Bounds, by: f32) -> Bounds {
    [b[0] - by, b[1] - by, b[2] + by, b[3] + by]
}

/// One layer drawn through a scratch target this frame.
#[derive(Clone, Debug)]
pub(crate) struct LayerDraw {
    /// Its members' draw list: a range of the frame's segments.
    pub segments: std::ops::Range<u32>,
    /// The scratch target it is drawn into.
    pub scratch: u32,
    /// The composite quad, in logical pixels.
    pub bounds: Bounds,
    pub style: OverlayLayerStyle,
    /// The composite's clip, in framebuffer pixels, and its mask index.
    pub clip_rect: [f32; 4],
    pub clip_index: f32,
}

impl LayerDraw {
    /// Whether the composite blends by formula against a copy of the
    /// destination.
    pub(crate) fn blends_by_formula(&self) -> bool {
        self.style.blend.reads_destination() || self.style.blend == OverlayBlend::Multiply
    }

    /// The pipeline the composite draws with when the destination cannot be
    /// read: a destination-reading mode as source-over.
    pub(crate) fn fallback_key(&self) -> LayerPipelineKey {
        let blend = if self.style.blend.reads_destination() {
            OverlayBlend::SourceOver
        } else {
            self.style.blend
        };
        LayerPipelineKey::Composite(blend)
    }

    /// The pipeline the composite draws with when it can read the
    /// destination.
    pub(crate) fn key(&self) -> LayerPipelineKey {
        if self.blends_by_formula() {
            LayerPipelineKey::CompositeDst
        } else {
            self.fallback_key()
        }
    }

    fn blurs(&self) -> bool {
        self.style.blur > 0.0
    }

    fn shadows(&self) -> bool {
        self.style.shadow.is_some_and(|s| s.colour.alpha() > 0.0)
    }

    /// What the composite reads of the destination, as `dst_read` bits.
    pub(crate) fn dst_read(&self) -> u8 {
        use crate::renderer::overlay_draw_order::dst_read;
        let mut bits = dst_read::NONE;
        if self.blends_by_formula() {
            bits |= dst_read::BLEND;
        }
        if self.style.backdrop.is_active() {
            bits |= dst_read::BACKDROP;
        }
        bits
    }
}

/// The composite bounds of a layer whose members cover `content`: grown by
/// its blur and its shadow, cut to its clip and the viewport.
pub(crate) fn composite_bounds(
    content: Bounds,
    style: &OverlayLayerStyle,
    clip: Option<Bounds>,
    viewport: [f32; 2],
    ppp: f32,
) -> Bounds {
    let mut b = expand(content, gaussian_reach(content_sigma(style.blur), ppp));
    if let Some(s) = style.shadow.filter(|s| s.colour.alpha() > 0.0) {
        let moved = [
            content[0] + s.offset[0],
            content[1] + s.offset[1],
            content[2] + s.offset[0],
            content[3] + s.offset[1],
        ];
        b = union(b, expand(moved, gaussian_reach(shadow_sigma(s.blur), ppp)));
    }
    if let Some(c) = clip {
        b = intersect(b, c);
    }
    intersect(b, [0.0, 0.0, viewport[0], viewport[1]])
}

/// A colour texture and its view.
type Target = (crate::gpu::Texture, crate::gpu::TextureView);

/// Three-pass blur into half-resolution textures: a bilinear downsample into
/// `a`, then horizontal into `b`, then vertical back into `a`.
struct BlurChain {
    a: Target,
    b: Target,
    h_uniform: crate::gpu::Buffer,
    v_uniform: crate::gpu::Buffer,
    h_bg: crate::gpu::BindGroup,
    v_bg: crate::gpu::BindGroup,
    downsample: Option<(crate::gpu::TextureView, crate::gpu::BindGroup)>,
    /// The spread and alpha mode the uniforms hold.
    written: Option<(f32, bool)>,
}

impl BlurChain {
    fn new(
        device: &crate::gpu::Device,
        resources: &crate::resources::DeviceResources,
        size: [u32; 2],
        format: crate::gpu::TextureFormat,
        allocations: &mut u32,
    ) -> Self {
        let half = [(size[0] / 2).max(1), (size[1] / 2).max(1)];
        let a = make_target(
            device,
            "overlay_layer_blur_a",
            half,
            format,
            false,
            allocations,
        );
        let b = make_target(
            device,
            "overlay_layer_blur_b",
            half,
            format,
            false,
            allocations,
        );
        let uniform = |label| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let h_uniform = uniform("overlay_layer_blur_h");
        let v_uniform = uniform("overlay_layer_blur_v");
        let bgl = resources
            .backdrop_blur
            .bgl
            .as_ref()
            .expect("blur pipeline built");
        let sampler = resources
            .backdrop_blur
            .sampler
            .as_ref()
            .expect("blur pipeline built");
        let bg = |label, view: &crate::gpu::TextureView, u: &crate::gpu::Buffer| {
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some(label),
                layout: bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::gpu::BindingResource::TextureView(view),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::gpu::BindingResource::Sampler(sampler),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: u.as_entire_binding(),
                    },
                ],
            })
        };
        let h_bg = bg("overlay_layer_blur_h_bg", &a.1, &h_uniform);
        let v_bg = bg("overlay_layer_blur_v_bg", &b.1, &v_uniform);
        Self {
            a,
            b,
            h_uniform,
            v_uniform,
            h_bg,
            v_bg,
            downsample: None,
            written: None,
        }
    }

    /// Write the tap spread (half-resolution texels) and whether alpha is
    /// blurred too.
    fn set(&mut self, queue: &crate::gpu::Queue, spread: f32, keep_alpha: bool) {
        if self.written == Some((spread, keep_alpha)) {
            return;
        }
        let bits = spread.to_bits();
        let keep = keep_alpha as u32;
        queue.write_buffer(
            &self.h_uniform,
            0,
            bytemuck::cast_slice(&[1u32, bits, keep, 0]),
        );
        queue.write_buffer(
            &self.v_uniform,
            0,
            bytemuck::cast_slice(&[0u32, bits, keep, 0]),
        );
        self.written = Some((spread, keep_alpha));
    }

    /// The downsample bind group for `source`, rebuilt when the source changes.
    fn source(
        &mut self,
        device: &crate::gpu::Device,
        resources: &crate::resources::DeviceResources,
        source: &crate::gpu::TextureView,
    ) {
        if self.downsample.as_ref().map(|(v, _)| v) == Some(source) {
            return;
        }
        let bgl = resources
            .post
            .dyn_res_upscale_bgl
            .as_ref()
            .expect("blit built");
        let sampler = resources
            .post
            .dyn_res_linear_sampler
            .as_ref()
            .expect("blit built");
        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("overlay_layer_downsample_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(source),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        self.downsample = Some((source.clone(), bg));
    }

    /// Encode the three passes over the downsample source set last, each
    /// scissored to `region` (full-resolution pixels) when given. The result
    /// is in `a`.
    fn run(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        resources: &crate::resources::DeviceResources,
        region: Option<[u32; 4]>,
    ) {
        let (Some(blit), Some(blur), Some((_, down_bg))) = (
            resources.post.dyn_res_upscale_pipeline.as_ref(),
            resources.backdrop_blur.pipeline.as_ref(),
            self.downsample.as_ref(),
        ) else {
            return;
        };
        let half = [self.a.0.width(), self.a.0.height()];
        let scissor = region.map(|r| {
            let x0 = (r[0] / 2).min(half[0]);
            let y0 = (r[1] / 2).min(half[1]);
            let x1 = r[2].div_ceil(2).min(half[0]);
            let y1 = r[3].div_ceil(2).min(half[1]);
            [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
        });
        if scissor.is_some_and(|s| s[2] == 0 || s[3] == 0) {
            return;
        }
        for (label, view, pipeline, bg) in [
            ("overlay_layer_downsample", &self.a.1, blit, down_bg),
            ("overlay_layer_blur_h", &self.b.1, blur, &self.h_bg),
            ("overlay_layer_blur_v", &self.a.1, blur, &self.v_bg),
        ] {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some(label),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(s) = scissor {
                pass.set_scissor_rect(s[0], s[1], s[2], s[3]);
            }
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn make_target(
    device: &crate::gpu::Device,
    label: &'static str,
    size: [u32; 2],
    format: crate::gpu::TextureFormat,
    copy_dst: bool,
    allocations: &mut u32,
) -> Target {
    *allocations += 1;
    let mut usage =
        crate::gpu::TextureUsages::RENDER_ATTACHMENT | crate::gpu::TextureUsages::TEXTURE_BINDING;
    if copy_dst {
        usage |= crate::gpu::TextureUsages::COPY_DST;
    }
    let texture = device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some(label),
        size: crate::gpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// One pooled scratch target.
struct LayerScratch {
    colour: Target,
    blur: Option<BlurChain>,
    shadow: Option<BlurChain>,
    composite: crate::resources::cached_bind_group::CachedBindGroup<[crate::gpu::TextureView; 5]>,
    /// The padded bounds of the layers drawn into it this frame.
    occupied: Vec<Bounds>,
    /// Whether a layer that blurs has it to itself this frame.
    exclusive: bool,
}

/// A viewport's layer targets, kept across frames, and the frame's layer
/// draws.
pub(crate) struct OverlayLayerTargets {
    size: [u32; 2],
    format: Option<crate::gpu::TextureFormat>,
    depth: Option<(crate::gpu::Texture, crate::gpu::TextureView)>,
    pool: Vec<LayerScratch>,
    /// A copy of the overlay target, taken at a break.
    snapshot: Option<Target>,
    /// The destination blurred, for backdrops.
    backdrop: Option<BlurChain>,
    /// The backdrop pipeline's group 0 over the blurred destination.
    item_backdrop: crate::resources::cached_bind_group::CachedBindGroup<crate::gpu::TextureView>,
    clip_bg: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        crate::gpu::TextureView,
    )>,
    instances: crate::renderer::overlay_buffers::GrowBuffer,
    /// The frame's composites, indexed by `OverlayDrawSource::Layer`.
    pub draws: Vec<LayerDraw>,
    pub instance_buf: Option<crate::gpu::Buffer>,
    /// Group 1 of the composite: clip shapes, viewport, shadow layers.
    pub composite_clip_bg: Option<crate::gpu::BindGroup>,
    /// The blur radius the frame's backdrops use.
    pub backdrop_radius: f32,
    /// Textures created by the last prepare.
    pub allocations: u32,
    /// Scratch targets the last prepare drew into.
    pub targets_used: u32,
}

impl OverlayLayerTargets {
    pub(crate) fn new() -> Self {
        Self {
            size: [0, 0],
            format: None,
            depth: None,
            pool: Vec::new(),
            snapshot: None,
            backdrop: None,
            item_backdrop: crate::resources::cached_bind_group::CachedBindGroup::new(),
            clip_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            instances: crate::renderer::overlay_buffers::GrowBuffer::vertex(
                "overlay_layer_instances",
            ),
            draws: Vec::new(),
            instance_buf: None,
            composite_clip_bg: None,
            backdrop_radius: 0.0,
            allocations: 0,
            targets_used: 0,
        }
    }

    /// The composite bind group (group 0) of scratch target `scratch`.
    pub(crate) fn composite_bg(&self, scratch: u32) -> Option<&crate::gpu::BindGroup> {
        self.pool.get(scratch as usize)?.composite.current()
    }

    /// The backdrop pipeline's group 0 over the blurred destination, when the
    /// frame has backdrops.
    pub(crate) fn item_backdrop_bg(&self) -> Option<&crate::gpu::BindGroup> {
        self.backdrop.as_ref()?;
        self.item_backdrop.current()
    }

    /// Whether `target` can be copied into the snapshot at a break: it has
    /// `COPY_SRC` and the snapshot's size and format.
    pub(crate) fn can_snapshot(&self, target: &crate::gpu::Texture) -> bool {
        self.snapshot.as_ref().is_some_and(|(s, _)| {
            target.usage().contains(crate::gpu::TextureUsages::COPY_SRC)
                && target.width() == s.width()
                && target.height() == s.height()
                && target.format() == s.format()
        })
    }

    /// Copy `region` (framebuffer pixels) of `target` into the snapshot and,
    /// with `blur`, blur it for the backdrops. Call between overlay passes.
    pub(crate) fn snapshot(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        resources: &crate::resources::DeviceResources,
        target: &crate::gpu::Texture,
        region: [u32; 4],
        blur: bool,
    ) {
        let Some((snap, _)) = &self.snapshot else {
            return;
        };
        let x1 = region[2].min(snap.width());
        let y1 = region[3].min(snap.height());
        if region[0] >= x1 || region[1] >= y1 {
            return;
        }
        encoder.copy_texture_to_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: crate::gpu::Origin3d {
                    x: region[0],
                    y: region[1],
                    z: 0,
                },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyTextureInfo {
                texture: snap,
                mip_level: 0,
                origin: crate::gpu::Origin3d {
                    x: region[0],
                    y: region[1],
                    z: 0,
                },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::Extent3d {
                width: x1 - region[0],
                height: y1 - region[1],
                depth_or_array_layers: 1,
            },
        );
        if blur {
            if let Some(chain) = &self.backdrop {
                chain.run(encoder, resources, Some(region));
            }
        }
    }

    /// Blur `source` (the scene, when the overlay target cannot be copied)
    /// into the backdrop texture. Returns whether there is a backdrop to draw.
    pub(crate) fn blur_scene(
        &mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        device: &crate::gpu::Device,
        resources: &crate::resources::DeviceResources,
        source: &crate::gpu::TextureView,
    ) -> bool {
        let Some(chain) = self.backdrop.as_mut() else {
            return false;
        };
        chain.source(device, resources, source);
        chain.run(encoder, resources, None);
        true
    }

    /// The snapshot region a break copies: its bounds, grown by the blur's
    /// reach when it blurs, in framebuffer pixels.
    pub(crate) fn break_region(&self, bounds: Bounds, blur: bool, ppp: f32) -> [u32; 4] {
        let b = if blur {
            expand(bounds, blur_reach(self.backdrop_radius, ppp))
        } else {
            bounds
        };
        let px = |v: f32| (v * ppp).max(0.0);
        [
            px(b[0]).floor() as u32,
            px(b[1]).floor() as u32,
            px(b[2]).ceil() as u32,
            px(b[3]).ceil() as u32,
        ]
    }
}

/// Everything `prepare_layer_targets` needs from the frame.
pub(crate) struct LayerFrame {
    /// The viewport's physical size, which every layer target takes.
    pub size: [u32; 2],
    pub ppp: f32,
    /// Whether the frame has a break that blurs, or a layer with a backdrop.
    pub needs_backdrop: bool,
    /// Whether the frame has any break.
    pub needs_snapshot: bool,
}

impl crate::renderer::ViewportRenderer {
    /// Allocate and fill this frame's layer targets: assign each composite a
    /// scratch target, draw the members into it, blur what blurs, write the
    /// composite instances, and make sure the snapshot and backdrop textures a
    /// break needs exist. Returns the command buffer that draws the members,
    /// which has to be submitted before the overlay pass.
    pub(crate) fn prepare_layer_targets(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ov: &mut crate::renderer::ViewportOverlay,
        lf: LayerFrame,
    ) -> Option<crate::gpu::CommandBuffer> {
        let lt = &mut ov.layer_targets;
        lt.allocations = 0;
        lt.targets_used = 0;
        lt.instance_buf = None;
        lt.composite_clip_bg = None;
        if lt.draws.is_empty() && !lf.needs_snapshot && !lf.needs_backdrop {
            return None;
        }
        let format = self.resources.target_format;
        if lt.size != lf.size || lt.format != Some(format) {
            lt.pool.clear();
            lt.depth = None;
            lt.snapshot = None;
            lt.backdrop = None;
            lt.size = lf.size;
            lt.format = Some(format);
        }
        self.resources.ensure_overlay_layer_resources(device, queue);
        self.resources.ensure_backdrop_blur_pipeline(device);
        self.resources.ensure_dyn_res_pipeline(device);
        for i in 0..lt.draws.len() {
            let d = &lt.draws[i];
            let keys = [
                Some(d.fallback_key()),
                Some(d.key()),
                d.style
                    .backdrop
                    .is_active()
                    .then_some(LayerPipelineKey::Backdrop),
            ];
            for key in keys.into_iter().flatten() {
                self.resources.ensure_overlay_layer_pipeline(device, key);
            }
        }
        let resources = &self.resources;
        let lt = &mut ov.layer_targets;
        let mut allocations = 0u32;

        // Scratch assignment.
        for s in &mut lt.pool {
            s.occupied.clear();
            s.exclusive = false;
        }
        let mut used = 0usize;
        for i in 0..lt.draws.len() {
            let d = &lt.draws[i];
            let padded = expand(d.bounds, 2.0);
            let alone = d.blurs() || d.shadows();
            let slot = if alone {
                None
            } else {
                lt.pool[..used]
                    .iter()
                    .position(|s| !s.exclusive && !s.occupied.iter().any(|o| overlaps(o, &padded)))
            };
            let slot = match slot {
                Some(s) => s,
                None => {
                    if used == lt.pool.len() {
                        lt.pool.push(LayerScratch {
                            colour: make_target(
                                device,
                                "overlay_layer_scratch",
                                lf.size,
                                format,
                                false,
                                &mut allocations,
                            ),
                            blur: None,
                            shadow: None,
                            composite: crate::resources::cached_bind_group::CachedBindGroup::new(),
                            occupied: Vec::new(),
                            exclusive: false,
                        });
                    }
                    used += 1;
                    used - 1
                }
            };
            let s = &mut lt.pool[slot];
            s.occupied.push(padded);
            s.exclusive |= alone;
            if d.blurs() {
                let chain = s.blur.get_or_insert_with(|| {
                    BlurChain::new(device, resources, lf.size, format, &mut allocations)
                });
                chain.set(
                    queue,
                    gaussian_spread(content_sigma(d.style.blur), lf.ppp),
                    true,
                );
                chain.source(device, resources, &s.colour.1);
            }
            if let Some(sh) = d.style.shadow.filter(|_| d.shadows()) {
                let chain = s.shadow.get_or_insert_with(|| {
                    BlurChain::new(device, resources, lf.size, format, &mut allocations)
                });
                chain.set(queue, gaussian_spread(shadow_sigma(sh.blur), lf.ppp), true);
                chain.source(device, resources, &s.colour.1);
            }
            lt.draws[i].scratch = slot as u32;
        }
        lt.targets_used = used as u32;

        // Snapshot and backdrop textures.
        if lf.needs_snapshot && lt.snapshot.is_none() {
            lt.snapshot = Some(make_target(
                device,
                "overlay_layer_snapshot",
                lf.size,
                format,
                true,
                &mut allocations,
            ));
        }
        if lf.needs_backdrop {
            let chain = lt.backdrop.get_or_insert_with(|| {
                BlurChain::new(device, resources, lf.size, format, &mut allocations)
            });
            // The backdrop keeps the frosted-glass blur's own spread.
            chain.set(queue, (lt.backdrop_radius / 2.0).max(1.0), false);
            if let Some((_, snap)) = &lt.snapshot {
                chain.source(device, resources, snap);
            }
            if let (Some(bgl), Some(sampler)) = (
                resources.overlay_shape.tex_bgl.as_ref(),
                resources.overlay_shape.tex_sampler.as_ref(),
            ) {
                let view = chain.a.1.clone();
                lt.item_backdrop.get(view.clone(), || {
                    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("overlay_backdrop_bg"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(&view),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: crate::gpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    })
                });
            }
        }
        lt.allocations = allocations;
        if lt.draws.is_empty() {
            return None;
        }

        // Composite bind groups.
        let (Some(bgl), Some(sampler), Some(blank)) = (
            resources.overlay_layer.bgl.as_ref(),
            resources.overlay_layer.sampler.as_ref(),
            resources.overlay_layer.blank.as_ref(),
        ) else {
            return None;
        };
        let dst = lt.snapshot.as_ref().map_or(blank, |(_, v)| v).clone();
        let backdrop = lt.backdrop.as_ref().map_or(blank, |c| &c.a.1).clone();
        for s in lt.pool[..used].iter_mut() {
            let views = [
                s.colour.1.clone(),
                s.blur.as_ref().map_or(blank, |c| &c.a.1).clone(),
                s.shadow.as_ref().map_or(blank, |c| &c.a.1).clone(),
                dst.clone(),
                backdrop.clone(),
            ];
            s.composite.get(views.clone(), || {
                let entry = |binding, view| crate::gpu::BindGroupEntry {
                    binding,
                    resource: crate::gpu::BindingResource::TextureView(view),
                };
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("overlay_layer_composite_bg"),
                    layout: bgl,
                    entries: &[
                        entry(0, &views[0]),
                        entry(1, &views[1]),
                        entry(2, &views[2]),
                        entry(3, &views[3]),
                        entry(4, &views[4]),
                        crate::gpu::BindGroupEntry {
                            binding: 5,
                            resource: crate::gpu::BindingResource::Sampler(sampler),
                        },
                    ],
                })
            });
        }

        // Group 1: clip shapes, viewport, shadow layers.
        let shadow_view = match ov.overlay_bindings.shape_shadow.view() {
            Some(v) => v,
            None => ov.overlay_bindings.shape_shadow.write_changed(
                device,
                queue,
                &[crate::resources::OverlayShadowLayerGpu {
                    colour: [0.0; 4],
                    params: [0.0; 4],
                    params2: [0.0, 1.0, 0.0, 0.0],
                }],
            ),
        };
        let (Some(clip_buf), Some(vp_buf), Some(clip_bgl)) = (
            ov.overlay_bindings.clip.buffer().cloned(),
            ov.overlay_viewport_buf.clone(),
            resources.overlay_shape.tex_clip_bgl.as_ref(),
        ) else {
            return None;
        };
        let lt = &mut ov.layer_targets;
        let key = (clip_buf.clone(), vp_buf.clone(), shadow_view.clone());
        lt.composite_clip_bg = Some(lt.clip_bg.get(key, || {
            device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("overlay_layer_clip_bg"),
                layout: clip_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::resources::overlay::overlay_shape::clip_binding(&clip_buf),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: vp_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: crate::gpu::BindingResource::TextureView(&shadow_view),
                    },
                ],
            })
        }));

        // Composite instances.
        let instances: Vec<OverlayLayerInstance> = lt.draws.iter().map(layer_instance).collect();
        lt.instance_buf = Some(lt.instances.write(device, queue, &instances));

        // Layer depth target for the member passes.
        if lt.depth.is_none() {
            let texture = device.create_texture(&crate::gpu::TextureDescriptor {
                label: Some("overlay_layer_depth"),
                size: crate::gpu::Extent3d {
                    width: lf.size[0].max(1),
                    height: lf.size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: crate::gpu::TextureDimension::D2,
                format: resources.scene_depth_format(),
                usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            lt.depth = Some((texture, view));
            lt.allocations += 1;
        }

        // Draw the members, then blur what blurs.
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("overlay_layer_encoder"),
        });
        let ov: &crate::renderer::ViewportOverlay = ov;
        let lt = &ov.layer_targets;
        let depth = &lt.depth.as_ref().expect("made above").1;
        for (slot, scratch) in lt.pool[..used].iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("overlay_layer_members"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: &scratch.colour.1,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
                        store: crate::gpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            for d in lt.draws.iter().filter(|d| d.scratch as usize == slot) {
                let segs =
                    &ov.overlay_draw_segments[d.segments.start as usize..d.segments.end as usize];
                crate::renderer::render::emit_overlay_segments(
                    resources,
                    ov,
                    &mut pass,
                    segs,
                    &crate::renderer::render::OverlayEmit::MEMBERS,
                );
            }
        }
        for d in &lt.draws {
            let s = &lt.pool[d.scratch as usize];
            if d.blurs() {
                if let Some(c) = &s.blur {
                    c.run(&mut encoder, resources, None);
                }
            }
            if d.shadows() {
                if let Some(c) = &s.shadow {
                    c.run(&mut encoder, resources, None);
                }
            }
        }
        Some(encoder.finish())
    }
}

/// The composite instance for `d`.
fn layer_instance(d: &LayerDraw) -> OverlayLayerInstance {
    use crate::resources::overlay::overlay_layer::{blend_code, layer_flags};
    let mut flags = 0u32;
    if d.blurs() {
        flags |= layer_flags::BLUR;
    }
    let matrix = d
        .style
        .colour_matrix
        .filter(|m| !m.is_identity())
        .inspect(|_| flags |= layer_flags::MATRIX)
        .unwrap_or_default();
    let (shadow_colour, shadow) = match d.style.shadow.filter(|_| d.shadows()) {
        Some(s) => {
            flags |= layer_flags::SHADOW;
            (
                s.colour.to_linear_rgba(),
                [s.offset[0], s.offset[1], 0.0, 0.0],
            )
        }
        None => ([0.0; 4], [0.0; 4]),
    };
    let bd = &d.style.backdrop;
    OverlayLayerInstance {
        rect: d.bounds,
        params: [
            d.style.opacity.clamp(0.0, 1.0),
            blend_code(d.style.blend),
            flags as f32,
            0.0,
        ],
        matrix: matrix.columns(),
        shadow_colour,
        shadow,
        backdrop: [
            bd.saturation,
            bd.brightness,
            bd.hue_shift,
            if bd.is_active() { 1.0 } else { 0.0 },
        ],
        clip_rect: d.clip_rect,
        clip: [d.clip_index, 0.0, 0.0, 0.0],
    }
}

/// Whether two layers' padded bounds keep them in separate targets; exposed
/// for the assignment test.
#[cfg(test)]
fn shares(a: Bounds, b: Bounds) -> bool {
    !overlaps(&expand(a, 2.0), &expand(b, 2.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Colour;
    use crate::renderer::types::{OverlayFill, OverlayFrame, OverlayShape, OverlayShapeItem};

    fn rect(x: f32) -> OverlayShapeItem {
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [x, 0.0],
            [10.0, 10.0],
        )
        .with_fill(OverlayFill::Solid(Colour::WHITE))
    }

    #[test]
    fn a_frame_without_layers_resolves_to_nothing() {
        let mut frame = OverlayFrame::default();
        frame.shapes.push(rect(0.0));
        let mut plan = LayerPlan::default();
        plan.build(&frame);
        assert!(plan.layers.is_empty());
        assert_eq!(plan.record_layer(Family::Shape, 0), NO_LAYER);
    }

    #[test]
    fn fold_rules() {
        let mut frame = OverlayFrame::default();
        // A faded single item: pushed.
        let m = frame.layer_mark();
        frame.shapes.push(rect(0.0));
        frame.push_layer(m, OverlayLayerStyle::default().with_opacity(0.5));
        // Two items at full opacity: pass-through.
        let m = frame.layer_mark();
        frame.shapes.push(rect(20.0));
        frame.shapes.push(rect(25.0));
        frame.push_layer(m, OverlayLayerStyle::default());
        // Two faded items: a target.
        let m = frame.layer_mark();
        frame.shapes.push(rect(40.0));
        frame.shapes.push(rect(45.0));
        frame.push_layer(m, OverlayLayerStyle::default().with_opacity(0.5));
        // An item outside every layer that reads the destination: its own
        // target.
        let mut d = rect(60.0);
        d.style.blend = OverlayBlend::Difference;
        frame.shapes.push(d);
        let mut plan = LayerPlan::default();
        plan.build(&frame);
        let modes: Vec<LayerMode> = plan.layers.iter().map(|l| l.mode).collect();
        assert_eq!(
            modes,
            vec![
                LayerMode::Push,
                LayerMode::Passthrough,
                LayerMode::Target,
                LayerMode::Target
            ]
        );
        let pushed = plan
            .adjusted_style(Family::Shape, 0, &frame.shapes[0].style)
            .unwrap();
        assert_eq!(pushed.opacity, 0.5);
        assert_eq!(plan.record_layer(Family::Shape, 0), NO_LAYER);
        assert_eq!(plan.record_layer(Family::Shape, 3), 2);
        // The destination-reading item draws source-over into its target, and
        // the target composites with its blend.
        let inside = plan
            .adjusted_style(Family::Shape, 5, &frame.shapes[5].style)
            .unwrap();
        assert_eq!(inside.blend, OverlayBlend::SourceOver);
        assert_eq!(plan.layers[3].style.blend, OverlayBlend::Difference);
    }

    #[test]
    fn disjoint_layers_share() {
        assert!(shares([0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]));
        assert!(!shares([0.0, 0.0, 10.0, 10.0], [11.0, 0.0, 30.0, 10.0]));
    }

    #[test]
    fn composite_bounds_grow_by_blur_and_shadow() {
        let style = OverlayLayerStyle::default().with_shadow(
            crate::renderer::types::ShadowLayer::new(Colour::BLACK, 0.0, [5.0, 5.0]),
        );
        let b = composite_bounds([10.0, 10.0, 20.0, 20.0], &style, None, [100.0, 100.0], 1.0);
        assert_eq!(b, [10.0, 10.0, 25.0, 25.0]);
        let clipped = composite_bounds(
            [10.0, 10.0, 20.0, 20.0],
            &style,
            Some([0.0, 0.0, 22.0, 22.0]),
            [100.0, 100.0],
            1.0,
        );
        assert_eq!(clipped, [10.0, 10.0, 22.0, 22.0]);
    }
}
