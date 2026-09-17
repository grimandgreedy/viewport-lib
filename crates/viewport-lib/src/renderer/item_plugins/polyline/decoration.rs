//! The polyline vector-quantity decoration: arrow glyphs generated from a
//! [`PolylineItem`]'s `node_vectors` and `edge_vectors`.
//!
//! This owns a second copy of the glyph render pipelines. The glyph item type
//! has its own pair inside its plugin, and the two are built from the same
//! `glyph.wgsl` against the same shared layouts; keeping a copy here is what
//! lets the two item types stay independent of each other. The copy is confined
//! to this module so it can be collapsed into a shared one later without
//! touching anything else.

use crate::plugin_api::ItemFrameContext;
use crate::renderer::PolylineItem;
use crate::resources::{
    DeviceResources, DualPipeline, GlyphGpuData, Vertex, VertexBufferLayoutExt,
};

/// The decoration's pipelines and this frame's draw data.
#[derive(Default)]
pub(super) struct Decoration {
    pipelines: Option<Pipelines>,
    frame: Vec<GlyphGpuData>,
}

struct Pipelines {
    solid: DualPipeline,
    wireframe: DualPipeline,
}

impl Decoration {
    pub(super) fn reset(&mut self) {
        self.pipelines = None;
        self.frame.clear();
    }

    pub(super) fn clear_frame(&mut self) {
        self.frame.clear();
    }

    /// Generate and upload the decoration for one polyline item, if it carries
    /// vector quantities. Builds the pipelines on the first item that needs
    /// them, so a scene of undecorated polylines compiles nothing.
    pub(super) fn add_for_item(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        ctx: &ItemFrameContext<'_>,
        item: &PolylineItem,
    ) {
        if item.node_vectors.is_empty() && item.edge_vectors.is_empty() {
            return;
        }
        let wireframe = ctx.wireframe_mode || item.settings.wireframe;
        let mut upload = |glyphs: crate::renderer::GlyphItem| {
            if glyphs.positions.is_empty() {
                return;
            }
            self.pipelines
                .get_or_insert_with(|| Pipelines::new(device, ctx.resources));
            self.frame.push(
                ctx.resources
                    .upload_glyph_set_per_frame(device, queue, &glyphs, wireframe),
            );
        };
        if !item.node_vectors.is_empty() {
            upload(crate::quantities::polyline_node_vectors_to_glyphs(item));
        }
        if !item.edge_vectors.is_empty() {
            upload(crate::quantities::polyline_edge_vectors_to_glyphs(item));
        }
    }

    pub(super) fn paint(&self, pass: &mut crate::gpu::RenderPass<'_>, is_hdr: bool) {
        let Some(pipelines) = &self.pipelines else {
            return;
        };
        for gpu in &self.frame {
            let pipeline = if gpu.wireframe {
                pipelines.wireframe.for_format(is_hdr)
            } else {
                pipelines.solid.for_format(is_hdr)
            };
            pass.set_pipeline(pipeline);
            crate::renderer::item_plugins::glyph::pipeline::draw_set(
                pass,
                gpu,
                0..gpu.instance_count,
            );
        }
    }
}

impl Pipelines {
    fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let shader = crate::resources::builders::wgsl_module(
            device,
            "glyph_shader",
            crate::resources::builders::wgsl_source!("glyph"),
        );
        let layout = crate::resources::builders::pipeline_layout(
            device,
            "glyph_pipeline_layout",
            &[
                resources.shared_bindings().group0_layout,
                &resources.glyph.bgl,
                &resources.glyph.instance_bgl,
            ],
        );
        let vertex_buffers = [Vertex::buffer_layout()];
        let solid = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "glyph_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &vertex_buffers,
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(crate::gpu::Face::Back),
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );
        let wireframe = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "glyph_wireframe_pipeline",
                layout: &layout,
                shader: &shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &vertex_buffers,
                blend: None,
                topology: crate::gpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: resources.sample_count,
                ldr_format: resources.target_format,
            },
        );
        Self { solid, wireframe }
    }
}
