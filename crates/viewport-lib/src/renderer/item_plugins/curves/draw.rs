//! The draw, pick and outline-mask hook bodies the three curve mesh item types
//! share.
//!
//! All three upload the same [`StreamtubeGpuData`] shape, so everything from
//! the frame-state assembly down to the sub-object resolve is identical; only
//! the render pipelines differ. Keeping the bodies here rather than in each
//! plugin means the three cannot drift apart in how they answer a pick.

use super::pipeline::{CurveFrame, CurveMeshGpu, CurvePickGpu, draw_mesh, draw_solid_indexed};
use super::store::StreamtubeGpuData;
use crate::plugin_api::{PaintContext, PickContext, PickPassContext};
use crate::renderer::{PickId, PickMask, SubObjectRef};
use crate::resources::HDR_COLOR_FORMAT;

/// Pixel radius of a world-space radius measured at the curve's first control
/// point, for the screen-space pick tolerance.
pub(super) fn radius_in_pixels(positions: &[[f32; 3]], world_r: f32, ctx: &PickContext<'_>) -> f32 {
    crate::plugin_api::pick_helpers::world_radius_in_pixels(
        glam::Vec3::from(positions[0]),
        world_r,
        ctx.view_proj,
        ctx.viewport_size,
    )
}

/// Assemble one item's frame state, building the pick and mask bind groups the
/// draw hooks need. The instance bind group is skipped entirely when the item
/// is neither pickable nor outlined, so an ordinary frame allocates nothing.
pub(super) fn build_frame(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    gpu: &CurveMeshGpu,
    gpu_data: StreamtubeGpuData,
    outlined: bool,
) -> CurveFrame {
    build_frame_with(device, queue, &gpu.pick, gpu_data, outlined)
}

/// The `build_frame` body, taking the pick state directly so the ribbon plugin
/// (whose render pipelines differ) can reuse it.
pub(super) fn build_frame_with(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    pick: &CurvePickGpu,
    gpu_data: StreamtubeGpuData,
    outlined: bool,
) -> CurveFrame {
    let wants_bind_group = gpu_data.pick_id != PickId::NONE || outlined;
    let instance_bind_group = wants_bind_group.then(|| {
        pick.instance_bind_group(
            device,
            queue,
            "curve_pick_instance_bg",
            gpu_data.model,
            gpu_data.pick_id,
        )
    });
    let node_bind_group = (gpu_data.pick_id != PickId::NONE)
        .then(|| pick.node_bind_group(device, gpu_data.node_pick_buffer.as_ref()))
        .flatten();
    CurveFrame {
        tri_segment: gpu_data.tri_segment.clone(),
        tri_strip: gpu_data.tri_strip.clone(),
        gpu: gpu_data,
        instance_bind_group,
        node_bind_group,
        outlined,
    }
}

/// The `paint` body for the streamtube and tube types, which share a pipeline
/// description.
pub(super) fn paint_curve_mesh(
    pass: &mut crate::gpu::RenderPass<'_>,
    ctx: &PaintContext<'_>,
    gpu: Option<&CurveMeshGpu>,
    frame: &[CurveFrame],
) {
    let Some(gpu) = gpu else { return };
    let is_hdr = ctx.target_format == HDR_COLOR_FORMAT;
    for entry in frame {
        let gd = &entry.gpu;
        if gd.index_count == 0 && gd.edge_index_count == 0 {
            continue;
        }
        let pipeline = if gd.wireframe {
            gpu.wireframe_pipeline.for_format(is_hdr)
        } else {
            gpu.pipeline.for_format(is_hdr)
        };
        pass.set_pipeline(pipeline);
        draw_mesh(pass, gd);
    }
}

/// The `outline_mask` body: draw every selected item's solid triangle mesh, so
/// the outline follows the swept silhouette rather than a bounding proxy.
pub(super) fn outline_mask_curve_mesh(
    pass: &mut crate::gpu::RenderPass<'_>,
    pick: Option<&CurvePickGpu>,
    frame: &[CurveFrame],
) {
    let Some(pick) = pick else { return };
    let mut bound = false;
    for entry in frame {
        if !entry.outlined || entry.gpu.index_count == 0 {
            continue;
        }
        let Some(bg) = &entry.instance_bind_group else {
            continue;
        };
        if !bound {
            pass.set_pipeline(&pick.mask_pipeline);
            bound = true;
        }
        pass.set_bind_group(1, bg, &[]);
        draw_solid_indexed(pass, &entry.gpu);
    }
}

/// The `render_pick` body: rasterise the solid mesh under the item's pick id.
/// A query whose finest curve level is `POLY_NODE` switches to the variant that
/// writes the nearest segment endpoint straight into the primitive channel.
pub(super) fn render_pick_curve_mesh(
    pass: &mut crate::gpu::RenderPass<'_>,
    ctx: &PickPassContext<'_>,
    pick: Option<&CurvePickGpu>,
    frame: &[CurveFrame],
) {
    if !ctx
        .mask
        .intersects(PickMask::OBJECT | PickMask::POLY_NODE | PickMask::SEGMENT | PickMask::STRIP)
    {
        return;
    }
    let Some(pick) = pick else { return };
    // Matches the resolve-side priority STRIP > SEGMENT > POLY_NODE: the node
    // variant only fires when nothing coarser was asked for.
    let writes_node = ctx.mask.intersects(PickMask::POLY_NODE)
        && !ctx.mask.intersects(PickMask::STRIP | PickMask::SEGMENT);
    for entry in frame {
        if entry.gpu.pick_id == PickId::NONE || entry.gpu.index_count == 0 {
            continue;
        }
        let Some(instance_bg) = &entry.instance_bind_group else {
            continue;
        };
        let node = writes_node
            .then(|| {
                entry
                    .node_bind_group
                    .as_ref()
                    .zip(pick.node_pipeline.as_ref())
            })
            .flatten();
        match node {
            Some((node_bg, node_pipeline)) => {
                pass.set_pipeline(node_pipeline);
                pass.set_bind_group(1, instance_bg, &[]);
                pass.set_bind_group(2, node_bg, &[]);
            }
            None => {
                pass.set_pipeline(&pick.pick_pipeline);
                pass.set_bind_group(1, instance_bg, &[]);
            }
        }
        draw_solid_indexed(pass, &entry.gpu);
    }
}

/// The `resolve_sub_object` body: map a read-back primitive channel to the
/// curve level the query asked for. The node variant already wrote the final
/// node index, so `POLY_NODE` passes it straight through.
pub(super) fn resolve_curve_sub_object(
    frame: &[CurveFrame],
    pick_id: PickId,
    primitive_index: u32,
    mask: PickMask,
) -> Option<SubObjectRef> {
    let entry = frame.iter().find(|f| f.gpu.pick_id == pick_id)?;
    if mask.intersects(PickMask::STRIP) {
        return Some(SubObjectRef::Strip(
            entry
                .tri_strip
                .get(primitive_index as usize)
                .copied()
                .unwrap_or(0),
        ));
    }
    if mask.intersects(PickMask::SEGMENT) {
        return Some(SubObjectRef::Segment(
            entry
                .tri_segment
                .get(primitive_index as usize)
                .copied()
                .unwrap_or(0),
        ));
    }
    mask.intersects(PickMask::POLY_NODE)
        .then_some(SubObjectRef::Point(primitive_index))
}
