//! Retained overlay geometry: compiled vertex buffers keyed by
//! [`OverlayGeometryId`], plus the per-draw instance struct that carries a
//! group's per-frame translate, opacity, and clip to the overlay shaders.

/// A compiled overlay group's GPU geometry.
///
/// Holds the text-pipeline vertex stream (`OverlayTextVertex`: polylines, vector
/// fills, and glyph runs) in a persistent buffer that lives until the group is
/// freed. The SDF-shape stream is added later; a group may then carry either or
/// both.
pub(crate) struct CompiledOverlay {
    /// Text-pipeline vertices in local logical-pixel space, uploaded once.
    pub vertex_buf: crate::gpu::Buffer,
    /// Number of vertices in `vertex_buf`.
    pub vertex_count: u32,
    /// The text stream's fill records (gradient fills on its labels, runs,
    /// polylines and vector shapes). `None` when it has none.
    pub text_fill_tex: Option<crate::gpu::TextureView>,
    /// GPU bytes held by `text_fill_tex`.
    pub text_fill_bytes: u64,
    /// GPU bytes held by the shape stream: `shape_instance_buf`, `shadow_tex`
    /// and `shape_fill_tex`.
    ///
    /// The entry's total charge lives on its store slot. Only the text stream is
    /// rebuilt on a re-emit, so the shape half is kept here rather than measured
    /// again to work out the new total.
    pub shape_bytes: u64,
    /// Analytic SDF shapes, one `OverlayShapeInstance` each, drawn through the
    /// shape pipeline. `None` when the group has no SDF shapes.
    pub shape_instance_buf: Option<crate::gpu::Buffer>,
    /// Number of instances in `shape_instance_buf`.
    pub shape_instance_count: u32,
    /// The stacked shadow-layer texture the SDF shapes reference. Kept alive
    /// alongside `shape_instance_buf`; always at least one (dummy) layer when the
    /// shape stream is present.
    pub shadow_tex: Option<crate::gpu::TextureView>,
    /// The SDF shapes' fill records. Present whenever the shape stream is.
    pub shape_fill_tex: Option<crate::gpu::TextureView>,
    /// Retained source items for re-emission, present only when the group carries
    /// glyphs. Glyph geometry bakes atlas UVs at a physical size, so it goes stale
    /// when the atlas grows or `pixels_per_point` changes; the renderer re-emits
    /// from this source on that event. `None` for polyline/vector-only groups,
    /// whose geometry is viewport-independent and never invalidates.
    pub source: Option<CompiledSource>,
    /// The anchor a compiled label resolves each frame. `Some` only for a group
    /// compiled from a `LabelItem`: the renderer resolves this to a screen origin
    /// (a viewport corner, or a world point projected through the camera) and adds
    /// it to the per-frame translate, and skips the group when a world anchor is
    /// culled. `None` for every other group, whose translate is taken as-is.
    pub anchor: Option<viewport_lib_types::overlay::OverlayOrigin>,
    /// The group's extent in its own local logical pixels, `(min, max)`, taken
    /// over every vertex it compiled. `None` for an empty group. This is what
    /// the group's alignment shifts against when it is anchored, the same
    /// way an item's extent box does.
    pub bounds: Option<([f32; 2], [f32; 2])>,
}

/// The source items a glyph-bearing group retains so its geometry can be
/// re-emitted when its baked atlas UVs go stale.
#[derive(Clone)]
pub(crate) struct CompiledSource {
    pub polylines: Vec<viewport_lib_types::overlay::OverlayPolylineItem>,
    pub vector_shapes: Vec<viewport_lib_types::overlay::OverlayShapeItem>,
    pub glyph_runs: Vec<viewport_lib_types::overlay::GlyphRunItem>,
    /// Labels retained for re-emission. A label lays its text out at compile time
    /// (glyphs, background box, and leader line, all on the text stream); a DPI or
    /// atlas change re-lays it out from here.
    pub labels: Vec<viewport_lib_types::overlay::LabelItem>,
    /// The atlas growth version the current geometry baked UVs against.
    pub baked_atlas_version: u64,
    /// The coverage atlas's version the current geometry baked vector-shape
    /// cell UVs against.
    pub baked_coverage_version: u64,
    /// The `pixels_per_point` the current geometry baked glyphs at.
    pub baked_ppp: f32,
}

/// Per-draw instance data shared by immediate and retained overlay draws.
///
/// Slot 0 is the identity used by every immediate draw (no transform, full
/// opacity, no clip); each retained group gets its own slot. The text pipeline
/// reads it as a per-instance vertex stream of five `vec4` attributes, so a
/// draw over instances `i..i + 1` sees slot `i`. The shape pipeline, whose
/// instances are the shapes themselves, reads the same record from a uniform
/// buffer at a dynamic offset of `i * INSTANCE_UNIFORM_STRIDE`.
///
/// `translate`, `linear` and `pivot` mirror the group's `OverlayTransform` and
/// are applied as `pivot + linear . (p - pivot) + translate`, the outer half of
/// the composition contract. An immediate item has no slot of its own, so its
/// transform is folded into the vertex stream at emit.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayInstance {
    /// Logical-pixel offset applied to the vertex position before pixel-to-NDC.
    pub translate: [f32; 2],
    /// Opacity multiplier applied to the fragment alpha.
    pub opacity: f32,
    /// Per-frame clip-mask index into the frame's clip-shape buffer, resolved from
    /// the group's `clip.mask`, or `-1` for no shaped clip. The shaders prefer this
    /// over the baked per-vertex clip index when it is set, so a retained group
    /// clips to a mask registered this frame.
    pub clip_index: f32,
    /// Outer clip bounding box in framebuffer pixels `[x0, y0, x1, y1]`; all-zero
    /// means no clip.
    pub clip_rect: [f32; 4],
    /// Colour multiplier applied to the group's colour each frame; identity
    /// `[1, 1, 1, 1]`. Lets a colour flash or fade ride the instance instead of the
    /// vertex stream.
    pub tint: [f32; 4],
    /// The linear part of the group transform, row-major `[a, b, c, d]` for
    /// `x' = a x + b y`, `y' = c x + d y`: rotation, scale, stretch and skew
    /// about `pivot`. Identity `[1, 0, 0, 1]`.
    pub linear: [f32; 4],
    /// Centre of the linear part, in logical pixels in the group's own space.
    pub pivot: [f32; 2],
    /// The length a unit distance covers after `linear`, its mean scale: the
    /// shape shader divides its anti-aliasing band by it so an edge stays one
    /// pixel wide whatever size the group is drawn at. Identity `1.0`.
    pub aa_scale: f32,
    pub _pad: f32,
}

// The five attributes cover the whole struct.
const _: () = assert!(std::mem::size_of::<OverlayInstance>() == 80);

/// Byte stride between instance records in the uniform copy of the instance
/// list the shape pipeline binds at a dynamic offset. Every tier guarantees a
/// dynamic-offset alignment no stricter than this.
pub(crate) const INSTANCE_UNIFORM_STRIDE: u64 = 256;

/// The instance stream's attributes in the text pipeline, after the text
/// vertex's seven.
pub(crate) const TEXT_INSTANCE_ATTRIBUTES: [crate::gpu::VertexAttribute; 5] =
    instance_attributes(7);

/// `[translate, opacity, clip_index]`, `clip_rect`, `tint`, `linear` and
/// `[pivot, aa_scale, pad]` at `first`, `first + 1`, ...
const fn instance_attributes(first: u32) -> [crate::gpu::VertexAttribute; 5] {
    const fn at(offset: u64, shader_location: u32) -> crate::gpu::VertexAttribute {
        crate::gpu::VertexAttribute {
            offset,
            shader_location,
            format: crate::gpu::VertexFormat::Float32x4,
        }
    }
    [
        at(0, first),
        at(16, first + 1),
        at(32, first + 2),
        at(48, first + 3),
        at(64, first + 4),
    ]
}

impl OverlayInstance {
    /// The instance stream's layout over `attributes`
    /// (`TEXT_INSTANCE_ATTRIBUTES`).
    pub(crate) fn buffer_layout(
        attributes: &'static [crate::gpu::VertexAttribute],
    ) -> crate::gpu::VertexBufferLayout<'static> {
        crate::gpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayInstance>() as crate::gpu::BufferAddress,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes,
        }
    }

    /// The identity instance immediate draws use: no offset, full opacity, no clip,
    /// no tint, no transform.
    pub const IDENTITY: OverlayInstance = OverlayInstance {
        translate: [0.0, 0.0],
        opacity: 1.0,
        clip_index: -1.0,
        clip_rect: [0.0, 0.0, 0.0, 0.0],
        tint: [1.0, 1.0, 1.0, 1.0],
        linear: [1.0, 0.0, 0.0, 1.0],
        pivot: [0.0, 0.0],
        aa_scale: 1.0,
        _pad: 0.0,
    };

    /// The group half of a transform as an instance's `linear`, `pivot` and
    /// `aa_scale`.
    pub(crate) fn set_transform(&mut self, t: &viewport_lib_types::overlay::OverlayTransform) {
        let m = t.linear();
        self.linear = [m[0][0], m[0][1], m[1][0], m[1][1]];
        self.pivot = t.pivot;
        self.aa_scale = t.uniform_scale_factor();
    }

    /// `instances` laid out one record per `INSTANCE_UNIFORM_STRIDE` bytes, for
    /// the uniform copy the shape pipeline binds at a dynamic offset.
    pub(crate) fn padded_uniform_bytes(instances: &[OverlayInstance]) -> Vec<u8> {
        let stride = INSTANCE_UNIFORM_STRIDE as usize;
        let mut out = vec![0u8; instances.len().max(1) * stride];
        for (i, inst) in instances.iter().enumerate() {
            out[i * stride..i * stride + 80].copy_from_slice(bytemuck::bytes_of(inst));
        }
        out
    }
}
