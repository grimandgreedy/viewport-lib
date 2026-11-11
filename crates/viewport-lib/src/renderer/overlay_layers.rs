//! Overlay layers: which items draw into a scratch target rather than the
//! frame, the scratch targets, and the passes that fill, blur and read them.
//!
//! Layers form a tree: a layer whose ranges lie inside another's is its
//! child, and an item belongs to the innermost layer naming it. A layer that
//! has nothing to do as a whole folds away before anything is drawn: a
//! pass-through layer leaves its members alone, a layer with one immediate
//! member pushes its opacity, blend and colour matrix onto that member, and a
//! layer whose one member is a layer, with only opacity of its own, passes
//! that opacity down. Only the rest (and every item whose blend reads the
//! destination) draws into a scratch target, which composites into the
//! target of the nearest layer above it that has one, or into the frame.
//!
//! A scratch target is the viewport's size, so members draw at their screen
//! positions with no change to their shaders and the composite reads them back
//! texel for texel. The targets are pooled per viewport and kept across
//! frames, and each frame hands them out by level: the number of composited
//! layers above a layer. Layers of one level whose padded bounds are disjoint
//! share a target, a layer that blurs its content or casts a shadow has one to
//! itself, and no target serves two levels in a frame, so a chain of layers N
//! deep holds N targets at once whatever its length.

use crate::renderer::overlay_draw_order::{
    Bounds, NO_LAYER, OverlayBreak, intersect, overlaps, union,
};
use crate::renderer::types::{
    OverlayBlend, OverlayClip, OverlayFrame, OverlayLayerStyle, OverlayStyle,
};
use crate::resources::overlay::overlay_layer::{LayerPipelineKey, OverlayLayerInstance};

/// How many layers deep the renderer composites: a layer with this many
/// layers above it draws flat.
pub(crate) const MAX_LAYER_DEPTH: u32 = 8;

/// How a layer draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LayerMode {
    /// Its members draw into the target above it unchanged.
    Passthrough,
    /// Its one member draws into the target above it with the layer's
    /// opacity, blend and colour matrix applied to it.
    Push,
    /// Its members draw into a scratch target that is composited as one.
    Target,
    /// Its members draw into the target above it, each with the layer's
    /// opacity, blend and colour matrix: a layer nested too deep to
    /// composite. A blend that reads the destination draws as source-over and
    /// backdrops are dropped.
    Flat,
}

/// A layer as the frame resolved it.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedLayer {
    pub style: OverlayLayerStyle,
    pub z_order: i32,
    pub clip: OverlayClip,
    pub mode: LayerMode,
    /// The layer it sits in, or [`NO_LAYER`].
    pub parent: u32,
    /// How many layers it sits in.
    pub depth: u32,
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

const FAMILIES: [Family; 6] = [
    Family::Shape,
    Family::Label,
    Family::GlyphRun,
    Family::Polyline,
    Family::Mesh,
    Family::Retained,
];

/// Every layer of a frame, and the innermost layer each item belongs to.
#[derive(Default)]
pub(crate) struct LayerPlan {
    pub layers: Vec<ResolvedLayer>,
    shapes: Vec<u32>,
    labels: Vec<u32>,
    glyph_runs: Vec<u32>,
    polylines: Vec<u32>,
    meshes: Vec<u32>,
    retained: Vec<u32>,
    /// Explicit layers ignored this frame because they partly overlap an
    /// earlier one: `(ignored, kept)` indices into `OverlayFrame::layers`.
    pub overlaps: Vec<(u32, u32)>,
    /// Layers drawn flat this frame because they sit too deep.
    pub flattened: u32,
}

/// A layer's member ranges, cut to the frame's list lengths, in [`FAMILIES`]
/// order.
type Span = [std::ops::Range<usize>; 6];

fn span_len(s: &Span) -> usize {
    s.iter().map(|r| r.len()).sum()
}

/// Whether every item `inner` names is in `outer`.
fn contains(outer: &Span, inner: &Span) -> bool {
    outer
        .iter()
        .zip(inner)
        .all(|(o, i)| i.is_empty() || (o.start <= i.start && i.end <= o.end))
}

/// Whether the two name an item in common.
fn spans_share(a: &Span, b: &Span) -> bool {
    a.iter()
        .zip(b)
        .any(|(x, y)| x.start.max(y.start) < x.end.min(y.end))
}

/// Fold a layer's style onto a member's, as a pushed or flat layer draws it.
fn fold_onto(s: &mut OverlayStyle, layer: &OverlayLayerStyle, flat: bool) {
    s.opacity *= layer.opacity;
    if layer.blend != OverlayBlend::SourceOver {
        s.blend = layer.blend;
    }
    if flat && s.blend.reads_destination() {
        s.blend = OverlayBlend::SourceOver;
    }
    if s.colour_matrix.is_none() {
        s.colour_matrix = layer.colour_matrix;
    }
    if flat {
        s.backdrop = Default::default();
    }
}

/// Whether a layer with `style` can push itself onto a lone member drawn with
/// `member`: the two blends and colour matrices compose.
fn can_push(style: &OverlayLayerStyle, member: &OverlayStyle) -> bool {
    !member.blend.reads_destination()
        && (style.blend == OverlayBlend::SourceOver || member.blend == OverlayBlend::SourceOver)
        && !(style.colour_matrix.is_some() && member.colour_matrix.is_some())
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

    /// The innermost layer item `index` of `family` belongs to, or
    /// [`NO_LAYER`].
    pub(crate) fn layer_of(&self, family: Family, index: usize) -> u32 {
        self.list(family).get(index).copied().unwrap_or(NO_LAYER)
    }

    /// The nearest layer from `l` up that draws through a scratch target, or
    /// [`NO_LAYER`] (the frame).
    pub(crate) fn target_of(&self, mut l: u32) -> u32 {
        while let Some(layer) = self.layers.get(l as usize) {
            if layer.mode == LayerMode::Target {
                return l;
            }
            l = layer.parent;
        }
        NO_LAYER
    }

    /// How many layers above `l` draw through a scratch target.
    pub(crate) fn level_of(&self, l: u32) -> u32 {
        let mut level = 0;
        let mut p = self.layers.get(l as usize).map_or(NO_LAYER, |x| x.parent);
        while let Some(layer) = self.layers.get(p as usize) {
            if layer.mode == LayerMode::Target {
                level += 1;
            }
            p = layer.parent;
        }
        level
    }

    /// The layer the item's draws are recorded into: the nearest layer it
    /// sits in that draws through a scratch target, [`NO_LAYER`] (the frame)
    /// when none does.
    pub(crate) fn record_layer(&self, family: Family, index: usize) -> u32 {
        self.target_of(self.layer_of(family, index))
    }

    /// Whether any layer draws through a scratch target.
    pub(crate) fn has_targets(&self) -> bool {
        self.layers.iter().any(|l| l.mode == LayerMode::Target)
    }

    /// The style item `index` of `family` draws with, when its layers change
    /// it: every pushed or flat layer between it and the target it draws
    /// into folded onto it, and, when that target took the item's
    /// destination-reading blend, the item drawn source-over without its
    /// backdrop. `None` when the item's own style stands.
    pub(crate) fn adjusted_style(
        &self,
        family: Family,
        index: usize,
        style: &OverlayStyle,
    ) -> Option<OverlayStyle> {
        let mut l = self.layer_of(family, index);
        let mut out: Option<OverlayStyle> = None;
        while let Some(layer) = self.layers.get(l as usize) {
            match layer.mode {
                LayerMode::Passthrough => {}
                LayerMode::Push | LayerMode::Flat => fold_onto(
                    out.get_or_insert_with(|| style.clone()),
                    &layer.style,
                    layer.mode == LayerMode::Flat,
                ),
                LayerMode::Target => {
                    let s = out.as_ref().unwrap_or(style);
                    if s.blend.reads_destination() {
                        let s = out.get_or_insert_with(|| style.clone());
                        s.blend = OverlayBlend::SourceOver;
                        s.backdrop = Default::default();
                    }
                    break;
                }
            }
            l = layer.parent;
        }
        out
    }

    /// Resolve `frame`'s layers. Cheap when the frame has none: the per-item
    /// lists stay empty and every lookup answers [`NO_LAYER`].
    pub(crate) fn build(&mut self, frame: &OverlayFrame) {
        self.layers.clear();
        self.overlaps.clear();
        self.flattened = 0;
        for f in FAMILIES {
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
        let lens = [
            frame.shapes.len(),
            frame.labels.len(),
            frame.glyph_runs.len(),
            frame.polylines.len(),
            frame.meshes.len(),
            frame.retained.len(),
        ];
        for (f, len) in FAMILIES.into_iter().zip(lens) {
            self.list_mut(f).resize(len, NO_LAYER);
        }

        // Explicit layers. One that shares items with an earlier kept layer
        // without either holding the other is ignored.
        let spans: Vec<Span> = frame
            .layers
            .iter()
            .map(|layer| {
                let m = &layer.members;
                let cut = |r: &std::ops::Range<usize>, len: usize| {
                    let end = r.end.min(len);
                    r.start.min(end)..end
                };
                [
                    cut(&m.shapes, lens[0]),
                    cut(&m.labels, lens[1]),
                    cut(&m.glyph_runs, lens[2]),
                    cut(&m.polylines, lens[3]),
                    cut(&m.meshes, lens[4]),
                    cut(&m.retained, lens[5]),
                ]
            })
            .collect();
        let mut kept: Vec<usize> = Vec::new();
        for j in 0..spans.len() {
            if span_len(&spans[j]) == 0 {
                continue;
            }
            let clash = kept.iter().copied().find(|&i| {
                spans_share(&spans[i], &spans[j])
                    && !contains(&spans[i], &spans[j])
                    && !contains(&spans[j], &spans[i])
            });
            match clash {
                Some(i) => self.overlaps.push((j as u32, i as u32)),
                None => kept.push(j),
            }
        }
        // The kept layers keep their order in the list, which is the order
        // their composites paint in at an equal `z_order`. Nesting is found
        // outer before inner: larger first and, of two naming the same items,
        // the later in the list. The layers holding a given layer then come
        // before it in that order, innermost last.
        let mut nesting: Vec<usize> = (0..kept.len()).collect();
        nesting.sort_by(|&a, &b| {
            span_len(&spans[kept[b]])
                .cmp(&span_len(&spans[kept[a]]))
                .then(kept[b].cmp(&kept[a]))
        });
        for &j in &kept {
            let layer = &frame.layers[j];
            self.layers.push(ResolvedLayer {
                style: layer.style.clone(),
                z_order: layer.z_order,
                clip: layer.clip,
                mode: LayerMode::Target,
                parent: NO_LAYER,
                depth: 0,
            });
        }
        for (t, &k) in nesting.iter().enumerate() {
            let span = &spans[kept[k]];
            if let Some(&p) = nesting[..t]
                .iter()
                .rev()
                .find(|&&p| contains(&spans[kept[p]], span))
            {
                self.layers[k].parent = p as u32;
                self.layers[k].depth = self.layers[p].depth + 1;
            }
            for (f, range) in FAMILIES.into_iter().zip(span.clone()) {
                for slot in &mut self.list_mut(f)[range] {
                    *slot = k as u32;
                }
            }
        }
        // A retained group with a layer style is a layer of its own, inside
        // whatever layer names the group.
        for (i, r) in frame.retained.iter().enumerate() {
            if let Some(style) = &r.layer {
                let parent = self.retained[i];
                self.retained[i] = self.push_node(style.clone(), r.z_order, r.clip, parent);
            }
        }

        // Each layer's direct members: items whose innermost layer it is, and
        // the layers directly inside it.
        let mut direct = vec![0usize; self.layers.len()];
        for f in FAMILIES {
            for (i, &l) in self.list(f).iter().enumerate() {
                let drawn = f != Family::Shape || frame.shapes[i].provides_mask.is_none();
                if drawn && let Some(c) = direct.get_mut(l as usize) {
                    *c += 1;
                }
            }
        }
        for l in 0..self.layers.len() {
            let p = self.layers[l].parent;
            if let Some(c) = direct.get_mut(p as usize) {
                *c += 1;
            }
        }

        // An item whose blend reads the destination is a one-item layer with
        // that blend, inside its own layer, unless it is that layer's only
        // member: then the layer takes the blend when it has none of its own.
        let implicit = |plan: &mut Self, family: Family, i: usize, style: &OverlayStyle, z| {
            if !reads(style) {
                return;
            }
            let p = plan.list(family)[i];
            if direct.get(p as usize) == Some(&1) {
                let layer = &mut plan.layers[p as usize];
                if layer.style.blend == OverlayBlend::SourceOver {
                    layer.style.blend = style.blend;
                }
                return;
            }
            let node = plan.push_node(
                OverlayLayerStyle::default().with_blend(style.blend),
                z,
                OverlayClip::default(),
                p,
            );
            plan.list_mut(family)[i] = node;
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
        self.resolve_modes(frame);
    }

    /// Append a layer inside `parent` and return its index.
    fn push_node(
        &mut self,
        style: OverlayLayerStyle,
        z_order: i32,
        clip: OverlayClip,
        parent: u32,
    ) -> u32 {
        let depth = self.layers.get(parent as usize).map_or(0, |p| p.depth + 1);
        self.layers.push(ResolvedLayer {
            style,
            z_order,
            clip,
            mode: LayerMode::Target,
            parent,
            depth,
        });
        (self.layers.len() - 1) as u32
    }

    /// Decide each layer's mode, innermost first, so a layer sees how its
    /// child layers resolved.
    fn resolve_modes(&mut self, frame: &OverlayFrame) {
        #[derive(Clone, Copy, Default)]
        struct Members {
            /// Drawn items whose innermost layer it is, and non-empty child
            /// layers.
            count: usize,
            /// The last direct item, or child layer, seen.
            item: Option<(Family, usize)>,
            child: Option<u32>,
            /// The one immediate item the layer draws, when it draws exactly
            /// one through no target of its own.
            lone: Option<(Family, usize)>,
        }
        let n = self.layers.len();
        let mut members = vec![Members::default(); n];
        for f in FAMILIES {
            for (i, &l) in self.list(f).iter().enumerate() {
                let drawn = f != Family::Shape || frame.shapes[i].provides_mask.is_none();
                if drawn && let Some(m) = members.get_mut(l as usize) {
                    m.count += 1;
                    m.item = Some((f, i));
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&l| std::cmp::Reverse(self.layers[l].depth));
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
        for l in order {
            let m = members[l];
            let lone = if m.count != 1 {
                None
            } else if let Some(c) = m.child {
                matches!(
                    self.layers[c as usize].mode,
                    LayerMode::Passthrough | LayerMode::Push
                )
                .then_some(members[c as usize].lone)
                .flatten()
            } else {
                m.item.filter(|(f, _)| *f != Family::Retained)
            };
            let layer = &self.layers[l];
            let style = &layer.style;
            let mut fold_into = None;
            let mode = if m.count == 0 || style.is_passthrough() {
                LayerMode::Passthrough
            } else if layer.depth >= MAX_LAYER_DEPTH {
                self.flattened += 1;
                LayerMode::Flat
            } else if style.needs_target() {
                LayerMode::Target
            } else if let Some(item) = lone
                && can_push(style, &self.style_below(item, l as u32, style_of(item)))
            {
                LayerMode::Push
            } else if let Some(c) = m.child.filter(|_| m.count == 1)
                && self.layers[c as usize].mode == LayerMode::Target
                && self.layers[c as usize].style.blend == OverlayBlend::SourceOver
                && style.blend == OverlayBlend::SourceOver
                && style.colour_matrix.is_none_or(|cm| cm.is_identity())
                && layer.clip == OverlayClip::default()
            {
                // Its one member composites alone: fading that composite is
                // the same as fading this layer's.
                fold_into = Some(c);
                LayerMode::Passthrough
            } else {
                LayerMode::Target
            };
            if let Some(c) = fold_into {
                let (opacity, z) = (style.opacity, layer.z_order);
                let child = &mut self.layers[c as usize];
                child.style.opacity *= opacity;
                child.z_order = z;
            }
            self.layers[l].mode = mode;
            members[l].lone = match mode {
                LayerMode::Passthrough | LayerMode::Push => lone,
                _ => None,
            };
            // Report to the parent. A pass-through layer around one layer
            // stands aside for it.
            let stand = match (mode, m.child) {
                (LayerMode::Passthrough, Some(c)) if m.count == 1 => c,
                _ => l as u32,
            };
            let p = self.layers[l].parent;
            if m.count > 0
                && let Some(pm) = members.get_mut(p as usize)
            {
                pm.count += 1;
                pm.child = Some(stand);
            }
        }
    }

    /// `style`, the style of `item`, with every pushed layer between the item
    /// and layer `stop` folded onto it.
    fn style_below(&self, item: (Family, usize), stop: u32, style: &OverlayStyle) -> OverlayStyle {
        let mut s = style.clone();
        let mut l = self.layer_of(item.0, item.1);
        while l != stop {
            let Some(layer) = self.layers.get(l as usize) else {
                break;
            };
            if layer.mode == LayerMode::Push {
                fold_onto(&mut s, &layer.style, false);
            }
            l = layer.parent;
        }
        s
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
    /// Where its members' list reads what is drawn in its target so far: a
    /// range of [`OverlayLayerTargets::breaks`].
    pub breaks: std::ops::Range<u32>,
    /// The scratch target it is drawn into.
    pub scratch: u32,
    /// How many composited layers it sits in.
    pub level: u32,
    /// What the composite covers, in the space's units.
    pub bounds: Bounds,
    /// The composite quad in logical pixels of the screen: `bounds`, or for
    /// a world layer their place on the screen.
    pub quad: Bounds,
    /// The drop shadow's offset in the quad's units.
    pub shadow_offset: [f32; 2],
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
/// its blur and its shadow, cut to its clip and to `area` (the viewport).
pub(crate) fn composite_bounds(
    content: Bounds,
    style: &OverlayLayerStyle,
    clip: Option<Bounds>,
    area: Bounds,
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
    intersect(b, area)
}

/// A colour texture and its view.
type Target = (crate::gpu::Texture, crate::gpu::TextureView);

/// Three-pass blur into half-resolution textures: a bilinear downsample into
/// `a`, then horizontal into `b`, then vertical back into `a`.
struct BlurChain {
    format: crate::gpu::TextureFormat,
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
            crate::gpu::TextureUsages::empty(),
            allocations,
        );
        let b = make_target(
            device,
            "overlay_layer_blur_b",
            half,
            format,
            crate::gpu::TextureUsages::empty(),
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
            format,
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
        let (Some((blit, blur)), Some((_, down_bg))) = (
            resources.overlay_blur_pipelines(self.format),
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
    copy: crate::gpu::TextureUsages,
    allocations: &mut u32,
) -> Target {
    *allocations += 1;
    let usage = crate::gpu::TextureUsages::RENDER_ATTACHMENT
        | crate::gpu::TextureUsages::TEXTURE_BINDING
        | copy;
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
    /// The level of the layers drawn into it this frame.
    level: u32,
}

/// Where a world layer lies on the screen, for the snapshot regions its
/// breaks copy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorldMap {
    /// Layer units to clip space.
    pub layer_to_clip: glam::Mat4,
    /// The target's size in device pixels.
    pub device_size: [f32; 2],
    /// Device pixels per logical pixel of the screen.
    pub screen_ppp: f32,
    /// Device pixels per layer unit.
    pub ppp: f32,
}

impl WorldMap {
    /// The device-pixel box `b` (layer units) covers, or the whole target
    /// when part of it is behind the camera.
    pub(crate) fn device_box(&self, b: Bounds) -> [f32; 4] {
        let mut out = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for (x, y) in [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])] {
            let c = self.layer_to_clip * glam::Vec4::new(x, y, 0.0, 1.0);
            if !(c.w > 1.0e-6) {
                return [0.0, 0.0, self.device_size[0], self.device_size[1]];
            }
            let px = (c.x / c.w * 0.5 + 0.5) * self.device_size[0];
            let py = (0.5 - c.y / c.w * 0.5) * self.device_size[1];
            out = [
                out[0].min(px),
                out[1].min(py),
                out[2].max(px),
                out[3].max(py),
            ];
        }
        out
    }

    /// The box in logical pixels of the screen.
    pub(crate) fn screen_box(&self, b: Bounds) -> Bounds {
        let d = self.device_box(b);
        let s = self.screen_ppp.max(1.0e-6);
        [d[0] / s, d[1] / s, d[2] / s, d[3] / s]
    }

    /// Offset `v` (layer units) at layer point `at`, in logical pixels of the
    /// screen.
    pub(crate) fn screen_offset(&self, at: [f32; 2], v: [f32; 2]) -> [f32; 2] {
        let p = |x: f32, y: f32| {
            let c = self.layer_to_clip * glam::Vec4::new(x, y, 0.0, 1.0);
            let w = c.w.max(1.0e-6);
            [
                (c.x / w * 0.5 + 0.5) * self.device_size[0],
                (0.5 - c.y / w * 0.5) * self.device_size[1],
            ]
        };
        let (a, b) = (p(at[0], at[1]), p(at[0] + v[0], at[1] + v[1]));
        let s = self.screen_ppp.max(1.0e-6);
        [(b[0] - a[0]) / s, (b[1] - a[1]) / s]
    }
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
    /// The destination blurred, for backdrops, forced opaque: the frame's
    /// target (the HDR scene clears alpha to zero) is covered where a
    /// backdrop draws.
    backdrop: Option<BlurChain>,
    /// A layer target blurred, for backdrops inside a layer, keeping its
    /// alpha: a backdrop over a part of the layer nothing has drawn into
    /// stays transparent.
    inner_backdrop: Option<BlurChain>,
    /// The backdrop pipeline's group 0 over each blurred copy.
    item_backdrop: crate::resources::cached_bind_group::CachedBindGroup<crate::gpu::TextureView>,
    item_backdrop_inner:
        crate::resources::cached_bind_group::CachedBindGroup<crate::gpu::TextureView>,
    clip_bg: crate::resources::cached_bind_group::CachedBindGroup<(
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        crate::gpu::TextureView,
    )>,
    instances: crate::renderer::overlay_buffers::GrowBuffer,
    /// The frame's composites, indexed by `OverlayDrawSource::Layer`,
    /// innermost first.
    pub draws: Vec<LayerDraw>,
    /// Where the composites' member lists read their targets.
    pub breaks: Vec<OverlayBreak>,
    /// For a world layer, where it lies on the screen.
    pub world: Option<WorldMap>,
    /// For a world layer, the screen viewport the composites draw over.
    composite_viewport: Option<crate::gpu::Buffer>,
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
            inner_backdrop: None,
            item_backdrop: crate::resources::cached_bind_group::CachedBindGroup::new(),
            item_backdrop_inner: crate::resources::cached_bind_group::CachedBindGroup::new(),
            clip_bg: crate::resources::cached_bind_group::CachedBindGroup::new(),
            instances: crate::renderer::overlay_buffers::GrowBuffer::vertex(
                "overlay_layer_instances",
            ),
            draws: Vec::new(),
            breaks: Vec::new(),
            world: None,
            composite_viewport: None,
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

    /// The same over the blurred layer target, for backdrops inside a layer.
    fn inner_backdrop_bg(&self) -> Option<&crate::gpu::BindGroup> {
        self.inner_backdrop.as_ref()?;
        self.item_backdrop_inner.current()
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
        let chain = self.backdrop.as_ref().filter(|_| blur);
        self.copy_and_blur(encoder, resources, target, region, chain);
    }

    /// The same for a break inside a layer's members, whose backdrops keep
    /// the target's alpha.
    fn snapshot_inner(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        resources: &crate::resources::DeviceResources,
        target: &crate::gpu::Texture,
        region: [u32; 4],
        blur: bool,
    ) {
        let chain = self.inner_backdrop.as_ref().filter(|_| blur);
        self.copy_and_blur(encoder, resources, target, region, chain);
    }

    fn copy_and_blur(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        resources: &crate::resources::DeviceResources,
        target: &crate::gpu::Texture,
        region: [u32; 4],
        chain: Option<&BlurChain>,
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
        if let Some(chain) = chain {
            chain.run(encoder, resources, Some(region));
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
        let d = match &self.world {
            Some(w) => w.device_box(b),
            None => [b[0] * ppp, b[1] * ppp, b[2] * ppp, b[3] * ppp],
        };
        let px = |v: f32| v.max(0.0);
        [
            px(d[0]).floor() as u32,
            px(d[1]).floor() as u32,
            px(d[2]).ceil() as u32,
            px(d[3]).ceil() as u32,
        ]
    }

    /// Whether any member list reads its target.
    pub(crate) fn has_breaks(&self) -> bool {
        !self.breaks.is_empty()
    }
}

/// Everything `prepare_layer_targets` needs from the frame.
pub(crate) struct LayerFrame {
    /// The viewport's physical size, which every layer target takes.
    pub size: [u32; 2],
    pub ppp: f32,
    /// Whether the frame's own list has a break that blurs.
    pub needs_backdrop: bool,
    /// Whether a layer's member list has a break that blurs.
    pub needs_inner_backdrop: bool,
    /// Whether the frame has any break.
    pub needs_snapshot: bool,
    /// The format of the layer targets and the snapshot: the format of the
    /// target the frame's overlays draw into.
    pub format: crate::gpu::TextureFormat,
    /// The colour format and sample count of the pass the outermost
    /// composites draw into.
    pub composite_pass: (crate::gpu::TextureFormat, u32),
    /// For a world layer: the pass its members draw into inside a layer
    /// target, and the screen viewport's logical size and pixels per point,
    /// which the composites draw over.
    pub world: Option<(
        crate::resources::overlay::world::WorldPassKey,
        [f32; 2],
        f32,
    )>,
}

/// Emit `ranges` of `ov`'s segments in one pass over `colour`, cleared first
/// when `clear` is set.
#[allow(clippy::too_many_arguments)]
fn member_pass(
    encoder: &mut crate::gpu::CommandEncoder,
    resources: &crate::resources::DeviceResources,
    ov: &crate::renderer::ViewportOverlay,
    colour: &crate::gpu::TextureView,
    depth: &crate::gpu::TextureView,
    clear: bool,
    ranges: &[std::ops::Range<u32>],
    emit: &crate::renderer::render::OverlayEmit<'_>,
) {
    let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
        #[cfg(any(wgpu29, wgpu30))]
        multiview_mask: None,
        label: Some("overlay_layer_members"),
        color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
            view: colour,
            resolve_target: None,
            ops: crate::gpu::Operations {
                load: if clear {
                    crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT)
                } else {
                    crate::gpu::LoadOp::Load
                },
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
    for r in ranges {
        let segs = &ov.overlay_draw_segments[r.start as usize..r.end as usize];
        crate::renderer::render::emit_overlay_segments(resources, ov, &mut pass, segs, emit);
    }
}

impl crate::renderer::ViewportRenderer {
    /// Report what `plan` could not draw as asked: layers ignored for
    /// overlapping an earlier one, and layers nested too deep.
    pub(crate) fn report_layer_plan(&self, plan: &LayerPlan) {
        for &(ignored, kept) in &plan.overlaps {
            self.resources.notices.report(
                crate::notice::NoticeKey::new(
                    crate::NoticeCategory::Item,
                    "overlay_layers_overlap",
                    [ignored as u64, kept as u64],
                ),
                crate::Severity::Warning,
                || crate::NoticeDetail::OverlayLayersOverlap { ignored, kept },
            );
        }
        if plan.flattened > 0 {
            let flattened = plan.flattened;
            self.resources.notices.report(
                crate::notice::NoticeKey::single(
                    crate::NoticeCategory::Item,
                    "overlay_layers_too_deep",
                ),
                crate::Severity::Warning,
                || crate::NoticeDetail::OverlayLayersTooDeep {
                    limit: MAX_LAYER_DEPTH,
                    flattened,
                },
            );
        }
    }

    /// Allocate and fill this frame's layer targets: assign each composite a
    /// scratch target, draw the members into it, blur what blurs, write the
    /// composite instances, and make sure the snapshot and backdrop textures a
    /// break needs exist. Returns the command buffer that draws the members,
    /// which has to be submitted before the overlay pass.
    ///
    /// The members draw innermost level first, so a layer's target is
    /// complete before the layer it sits in composites it. A member list that
    /// reads its target ends its pass at each break, copies what is drawn so
    /// far into the snapshot (blurring it for a backdrop) and resumes.
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
            // Member breaks imply composites, so `draws` is not empty then.
            return None;
        }
        let format = lf.format;
        if lt.size != lf.size || lt.format != Some(format) {
            lt.pool.clear();
            lt.depth = None;
            lt.snapshot = None;
            lt.backdrop = None;
            lt.inner_backdrop = None;
            lt.size = lf.size;
            lt.format = Some(format);
        }
        self.resources.ensure_overlay_layer_resources(device, queue);
        self.resources.ensure_backdrop_blur_pipeline(device);
        self.resources.ensure_dyn_res_pipeline(device);
        self.resources.ensure_overlay_blur_pipelines(device, format);
        for i in 0..lt.draws.len() {
            let d = &lt.draws[i];
            let (pass_format, samples) = if d.level == 0 {
                lf.composite_pass
            } else {
                (format, 1)
            };
            let keys = [
                Some(d.fallback_key()),
                Some(d.key()),
                d.style
                    .backdrop
                    .is_active()
                    .then_some(LayerPipelineKey::Backdrop),
            ];
            for key in keys.into_iter().flatten() {
                self.resources
                    .ensure_overlay_layer_pipeline_in(device, key, pass_format, samples);
            }
        }
        let resources = &self.resources;
        let lt = &mut ov.layer_targets;
        let mut allocations = 0u32;

        // Scratch assignment: within a level, layers with disjoint padded
        // bounds share a target; no target serves two levels.
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
                lt.pool[..used].iter().position(|s| {
                    s.level == d.level
                        && !s.exclusive
                        && !s.occupied.iter().any(|o| overlaps(o, &padded))
                })
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
                                crate::gpu::TextureUsages::COPY_SRC,
                                &mut allocations,
                            ),
                            blur: None,
                            shadow: None,
                            composite: crate::resources::cached_bind_group::CachedBindGroup::new(),
                            occupied: Vec::new(),
                            exclusive: false,
                            level: 0,
                        });
                    }
                    used += 1;
                    used - 1
                }
            };
            let s = &mut lt.pool[slot];
            s.occupied.push(padded);
            s.exclusive |= alone;
            s.level = d.level;
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
                crate::gpu::TextureUsages::COPY_DST,
                &mut allocations,
            ));
        }
        let spread = (lt.backdrop_radius / 2.0).max(1.0);
        for inner in [false, true] {
            let needed = if inner {
                lf.needs_inner_backdrop
            } else {
                lf.needs_backdrop
            };
            if !needed {
                continue;
            }
            let (chain, cache) = if inner {
                (&mut lt.inner_backdrop, &mut lt.item_backdrop_inner)
            } else {
                (&mut lt.backdrop, &mut lt.item_backdrop)
            };
            let chain = chain.get_or_insert_with(|| {
                BlurChain::new(device, resources, lf.size, format, &mut allocations)
            });
            // The backdrop keeps the frosted-glass blur's own spread.
            chain.set(queue, spread, inner);
            if let Some((_, snap)) = &lt.snapshot {
                chain.source(device, resources, snap);
            }
            if let (Some(bgl), Some(sampler)) = (
                resources.overlay_shape.tex_bgl.as_ref(),
                resources.overlay_shape.tex_sampler.as_ref(),
            ) {
                let view = chain.a.1.clone();
                cache.get(view.clone(), || {
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
        let inner_backdrop = lt.inner_backdrop.as_ref().map_or(blank, |c| &c.a.1).clone();
        for s in lt.pool[..used].iter_mut() {
            // A layer inside another draws its backdrop from the outer
            // layer's target.
            let views = [
                s.colour.1.clone(),
                s.blur.as_ref().map_or(blank, |c| &c.a.1).clone(),
                s.shadow.as_ref().map_or(blank, |c| &c.a.1).clone(),
                dst.clone(),
                if s.level > 0 {
                    inner_backdrop.clone()
                } else {
                    backdrop.clone()
                },
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

        // Group 1: clip shapes, viewport, shadow layers. A world layer's
        // composites draw over the screen viewport, not the layer's.
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
        let vp_buf = match lf.world {
            Some((_, screen, screen_ppp)) => {
                let buf = ov
                    .layer_targets
                    .composite_viewport
                    .get_or_insert_with(|| {
                        device.create_buffer(&crate::gpu::BufferDescriptor {
                            label: Some("overlay_layer_world_viewport"),
                            size: 16,
                            usage: crate::gpu::BufferUsages::UNIFORM
                                | crate::gpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        })
                    })
                    .clone();
                queue.write_buffer(
                    &buf,
                    0,
                    bytemuck::cast_slice(&[screen[0], screen[1], screen_ppp, 0.0]),
                );
                Some(buf)
            }
            None => ov.overlay_viewport_buf.clone(),
        };
        let (Some(clip_buf), Some(vp_buf), Some(clip_bgl)) = (
            ov.overlay_bindings.clip.buffer().cloned(),
            vp_buf,
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

        // Draw the members, innermost level first, then blur what blurs.
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("overlay_layer_encoder"),
        });
        let ov: &crate::renderer::ViewportOverlay = ov;
        let lt = &ov.layer_targets;
        let depth = &lt.depth.as_ref().expect("made above").1;
        let emit = crate::renderer::render::OverlayEmit {
            dst: true,
            backdrop: lt.inner_backdrop_bg(),
            world: lf.world.map(|(key, _, _)| key),
            masks: None,
        };
        let top = lt.draws.iter().map(|d| d.level).max().unwrap_or(0);
        let mut ranges: Vec<std::ops::Range<u32>> = Vec::new();
        for level in (0..=top).rev() {
            for (slot, scratch) in lt.pool[..used].iter().enumerate() {
                if scratch.level != level {
                    continue;
                }
                ranges.clear();
                let mut clear = true;
                for d in lt.draws.iter().filter(|d| d.scratch as usize == slot) {
                    let mut start = d.segments.start;
                    for b in &lt.breaks[d.breaks.start as usize..d.breaks.end as usize] {
                        ranges.push(start..b.segment);
                        member_pass(
                            &mut encoder,
                            resources,
                            ov,
                            &scratch.colour.1,
                            depth,
                            clear,
                            &ranges,
                            &emit,
                        );
                        ranges.clear();
                        clear = false;
                        let region = lt.break_region(b.bounds, b.blur, lf.ppp);
                        lt.snapshot_inner(
                            &mut encoder,
                            resources,
                            &scratch.colour.0,
                            region,
                            b.blur,
                        );
                        start = b.segment;
                    }
                    ranges.push(start..d.segments.end);
                }
                member_pass(
                    &mut encoder,
                    resources,
                    ov,
                    &scratch.colour.1,
                    depth,
                    clear,
                    &ranges,
                    &emit,
                );
            }
            for d in lt.draws.iter().filter(|d| d.level == level) {
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
                [d.shadow_offset[0], d.shadow_offset[1], 0.0, 0.0],
            )
        }
        None => ([0.0; 4], [0.0; 4]),
    };
    let bd = &d.style.backdrop;
    OverlayLayerInstance {
        rect: d.quad,
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

    /// Two items, then `inner` (pushed by the caller between the marks), as
    /// one layer drawn with `style`.
    fn wrap(
        frame: &mut OverlayFrame,
        x: f32,
        style: OverlayLayerStyle,
        inner: impl FnOnce(&mut OverlayFrame),
    ) {
        let m = frame.layer_mark();
        frame.shapes.push(rect(x));
        frame.shapes.push(rect(x + 5.0));
        inner(frame);
        frame.push_layer(m, style);
    }

    #[test]
    fn nested_layers_form_a_tree() {
        let mut frame = OverlayFrame::default();
        let faded = OverlayLayerStyle::default().with_opacity(0.5);
        wrap(
            &mut frame,
            0.0,
            OverlayLayerStyle::default().with_blur(2.0),
            |f| {
                wrap(f, 10.0, faded.clone(), |f| {
                    // A pass-through layer inside: its items draw into the faded
                    // layer's target.
                    wrap(f, 20.0, OverlayLayerStyle::default(), |_| {});
                });
            },
        );
        let mut plan = LayerPlan::default();
        plan.build(&frame);
        // The list order is the frame's: innermost pushed first.
        let modes: Vec<LayerMode> = plan.layers.iter().map(|l| l.mode).collect();
        assert_eq!(
            modes,
            vec![LayerMode::Passthrough, LayerMode::Target, LayerMode::Target]
        );
        assert_eq!(plan.layers[0].parent, 1);
        assert_eq!(plan.layers[1].parent, 2);
        assert_eq!(plan.layers[2].parent, NO_LAYER);
        assert_eq!(plan.layers[0].depth, 2);
        // The blurred layer's own items, the faded layer's, and the
        // pass-through layer's, which record into the faded one.
        assert_eq!(plan.record_layer(Family::Shape, 0), 2);
        assert_eq!(plan.record_layer(Family::Shape, 2), 1);
        assert_eq!(plan.record_layer(Family::Shape, 4), 1);
        assert_eq!(plan.target_of(plan.layers[1].parent), 2);
        assert_eq!(plan.level_of(1), 1);
        assert_eq!(plan.level_of(2), 0);
        assert!(plan.overlaps.is_empty());
    }

    #[test]
    fn nested_single_member_layers_fold() {
        let mut frame = OverlayFrame::default();
        // A faded single item inside a blurred group: pushed onto the item,
        // which draws into the group's target.
        wrap(
            &mut frame,
            0.0,
            OverlayLayerStyle::default().with_blur(2.0),
            |f| {
                let m = f.layer_mark();
                f.shapes.push(rect(10.0));
                f.push_layer(m, OverlayLayerStyle::default().with_opacity(0.5));
            },
        );
        // A faded layer whose one member is a blurred layer: the opacity
        // passes down and the faded layer draws through no target.
        let m = frame.layer_mark();
        wrap(
            &mut frame,
            40.0,
            OverlayLayerStyle::default().with_blur(2.0),
            |_| {},
        );
        frame.push_layer(m, OverlayLayerStyle::default().with_opacity(0.25));
        let mut plan = LayerPlan::default();
        plan.build(&frame);
        let modes: Vec<LayerMode> = plan.layers.iter().map(|l| l.mode).collect();
        assert_eq!(
            modes,
            vec![
                LayerMode::Push,
                LayerMode::Target,
                LayerMode::Target,
                LayerMode::Passthrough
            ]
        );
        assert_eq!(plan.record_layer(Family::Shape, 2), 1);
        let pushed = plan
            .adjusted_style(Family::Shape, 2, &frame.shapes[2].style)
            .unwrap();
        assert_eq!(pushed.opacity, 0.5);
        assert_eq!(plan.layers[2].style.opacity, 0.25);
    }

    #[test]
    fn a_destination_read_inside_a_layer_is_a_layer_of_its_own() {
        let mut frame = OverlayFrame::default();
        let mut diff = rect(2.0);
        diff.style.blend = OverlayBlend::Difference;
        let m = frame.layer_mark();
        frame.shapes.push(rect(0.0));
        frame.shapes.push(diff.clone());
        frame.push_layer(m, OverlayLayerStyle::default().with_opacity(0.5));
        // Alone in a layer, it lends the layer its blend instead.
        let m = frame.layer_mark();
        frame.shapes.push(diff);
        frame.push_layer(m, OverlayLayerStyle::default().with_opacity(0.5));
        let mut plan = LayerPlan::default();
        plan.build(&frame);
        assert_eq!(plan.layers.len(), 3);
        assert_eq!(plan.layers[2].parent, 0);
        assert_eq!(plan.layers[2].style.blend, OverlayBlend::Difference);
        assert_eq!(plan.layers[2].mode, LayerMode::Target);
        assert_eq!(plan.record_layer(Family::Shape, 1), 2);
        assert_eq!(plan.layers[1].style.blend, OverlayBlend::Difference);
        assert_eq!(plan.record_layer(Family::Shape, 2), 1);
        for i in [1, 2] {
            let s = plan
                .adjusted_style(Family::Shape, i, &frame.shapes[i].style)
                .unwrap();
            assert_eq!(s.blend, OverlayBlend::SourceOver);
        }
    }

    #[test]
    fn partly_overlapping_layers_keep_the_earlier() {
        let mut frame = OverlayFrame::default();
        for i in 0..4 {
            frame.shapes.push(rect(i as f32 * 20.0));
        }
        let faded = OverlayLayerStyle::default().with_opacity(0.5);
        frame.layers.push(crate::renderer::types::OverlayLayer::new(
            crate::renderer::types::OverlayLayerMembers::default().with_shapes(0..3),
            faded.clone(),
        ));
        frame.layers.push(crate::renderer::types::OverlayLayer::new(
            crate::renderer::types::OverlayLayerMembers::default().with_shapes(2..4),
            faded,
        ));
        let mut plan = LayerPlan::default();
        plan.build(&frame);
        assert_eq!(plan.overlaps, vec![(1, 0)]);
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.layer_of(Family::Shape, 2), 0);
        assert_eq!(plan.layer_of(Family::Shape, 3), NO_LAYER);
    }

    #[test]
    fn layers_past_the_depth_limit_draw_flat() {
        fn nest(f: &mut OverlayFrame, left: u32) {
            if left == 0 {
                return;
            }
            wrap(
                f,
                left as f32,
                OverlayLayerStyle::default().with_opacity(0.9),
                |f| nest(f, left - 1),
            );
        }
        let mut frame = OverlayFrame::default();
        nest(&mut frame, MAX_LAYER_DEPTH + 2);
        let mut plan = LayerPlan::default();
        plan.build(&frame);
        assert_eq!(plan.flattened, 2);
        let flat: Vec<u32> = plan
            .layers
            .iter()
            .filter(|l| l.mode == LayerMode::Flat)
            .map(|l| l.depth)
            .collect();
        assert_eq!(flat, vec![MAX_LAYER_DEPTH + 1, MAX_LAYER_DEPTH]);
        // The deepest items draw into the deepest composited target, faded by
        // both flat layers.
        let deepest = frame.shapes.len() - 1;
        let s = plan
            .adjusted_style(Family::Shape, deepest, &frame.shapes[deepest].style)
            .unwrap();
        assert!((s.opacity - 0.81).abs() < 1.0e-6);
        let target = plan.record_layer(Family::Shape, deepest);
        assert_eq!(plan.layers[target as usize].depth, MAX_LAYER_DEPTH - 1);
        assert_eq!(plan.level_of(target), MAX_LAYER_DEPTH - 1);
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
        let b = composite_bounds(
            [10.0, 10.0, 20.0, 20.0],
            &style,
            None,
            [0.0, 0.0, 100.0, 100.0],
            1.0,
        );
        assert_eq!(b, [10.0, 10.0, 25.0, 25.0]);
        let clipped = composite_bounds(
            [10.0, 10.0, 20.0, 20.0],
            &style,
            Some([0.0, 0.0, 22.0, 22.0]),
            [0.0, 0.0, 100.0, 100.0],
            1.0,
        );
        assert_eq!(clipped, [10.0, 10.0, 22.0, 22.0]);
    }
}
