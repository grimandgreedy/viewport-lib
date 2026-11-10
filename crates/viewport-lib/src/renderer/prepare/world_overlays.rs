//! Prepare for world overlay layers: each layer's frame runs through the
//! overlay prepare in a space of world units on its plane, with the device
//! pixels one unit covers at the camera's zoom standing in for
//! `pixels_per_point`.

use super::viewport_overlays::OverlaySpace;
use super::*;
use crate::resources::overlay::world::{WORLD_VIEWPORT_BYTES, WorldPassKey, WorldPipe};

/// How a world layer maps to the screen this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorldLayerView {
    /// Layer units to clip space, relative to the world origin.
    pub layer_to_clip: glam::Mat4,
    /// Device pixels one layer unit covers where the view's centre ray meets
    /// the layer.
    pub ppp: f32,
}

/// The view of a layer whose units map to clip space by `layer_to_clip`, on
/// a target `device_size` device pixels across. `None` when the layer is
/// seen edge-on or its centre point is behind the camera.
///
/// The scale is the square root of the area one layer unit covers on screen
/// at the layer point under the viewport's centre: exact for an orthographic
/// camera looking straight at the plane, where a unit covers a square, and the
/// mean of the two axes' scales under a tilted or perspective view.
pub(crate) fn world_layer_view(
    layer_to_clip: glam::Mat4,
    device_size: [f32; 2],
) -> Option<WorldLayerView> {
    // Clip x and y of layer point (u, v) on the plane are rows 0 and 1 of the
    // map applied to (u, v, 0, 1). The point under the centre has both zero.
    let r0 = layer_to_clip.row(0);
    let r1 = layer_to_clip.row(1);
    let r3 = layer_to_clip.row(3);
    let det = r0.x * r1.y - r0.y * r1.x;
    if det.abs() < 1.0e-20 {
        return None;
    }
    let u = (-r0.w * r1.y + r1.w * r0.y) / det;
    let v = (-r1.w * r0.x + r0.w * r1.x) / det;
    let w = r3.x * u + r3.y * v + r3.w;
    if !(w > 1.0e-12) {
        return None;
    }
    // There clip x and y are zero, so NDC moves by a row over w per unit,
    // and a device pixel is two NDC units over the target's size.
    let sx = device_size[0] * 0.5 / w;
    let sy = device_size[1] * 0.5 / w;
    let area = (det * sx * sy).abs();
    let ppp = area.sqrt();
    (ppp.is_finite() && ppp > 0.0).then_some(WorldLayerView { layer_to_clip, ppp })
}

impl ViewportRenderer {
    /// The pass the world layers draw into on the path `frame` takes: the
    /// read-only-depth pass on the HDR path, the scene pass on the LDR path.
    fn world_overlay_pass(&self, frame: &FrameData) -> WorldPassKey {
        if self.draws_hdr(frame) && self.resources.scene_tier.is_built() {
            WorldPassKey {
                format: crate::resources::HDR_COLOR_FORMAT,
                samples: self.resources.sample_count,
                depth_test: false,
            }
        } else {
            WorldPassKey {
                format: self.resources.target_format,
                samples: 1,
                depth_test: false,
            }
        }
    }

    /// Prepare `frame.scene.world_overlays` for its viewport.
    pub(super) fn prepare_world_overlays(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        sink: &mut crate::renderer::SubmitSink,
    ) {
        let vp_idx = frame.camera.viewport_index;
        let mut worlds = std::mem::take(&mut self.viewport_slots[vp_idx].world_overlays);
        worlds.sequence.clear();
        worlds.pass = None;
        let layers = &frame.scene.world_overlays;
        if layers.is_empty() {
            self.viewport_slots[vp_idx].world_overlays = worlds;
            return;
        }
        let _span =
            tracing::debug_span!(target: "viewport_lib::frame", "world_overlay_prepare").entered();
        // The screen overlays' counters are published before this runs, so a
        // world layer's draws and tessellations are kept out of them.
        let counters = self.overlay_retained_counters;
        let tessellations = self.overlay_tess_cache.tessellations;

        let pass = self.world_overlay_pass(frame);
        let screen_ppp = frame.camera.pixels_per_point.max(1.0e-6);
        let device_size = [
            frame.camera.viewport_size[0] * screen_ppp,
            frame.camera.viewport_size[1] * screen_ppp,
        ];
        let world_origin = self.origin.origin();
        let view_proj = self.origin.camera(&frame.camera.render_camera).view_proj();
        while worlds.layers.len() < layers.len() {
            worlds.layers.push(crate::renderer::WorldOverlayDraw {
                ov: crate::renderer::ViewportOverlay::new(),
                order: 0,
                depth_test: false,
            });
        }
        worlds.layers.truncate(layers.len());

        for (i, layer) in layers.iter().enumerate() {
            let layer_to_clip = view_proj * layer.layer_to_world(world_origin);
            let Some(view) = world_layer_view(layer_to_clip, device_size) else {
                continue;
            };
            let (anchor_view, anchor_proj) = layer.anchor_view_projection(world_origin);
            let space = OverlaySpace {
                overlays: &layer.frame,
                size: layer.anchor_extent(),
                ppp: view.ppp,
                view: anchor_view,
                proj: anchor_proj,
                world_origin,
                axes: None,
                world: true,
                screen_ppp,
            };
            let draw = &mut worlds.layers[i];
            draw.order = layer.order;
            draw.depth_test = layer.depth_test;
            let ov = &mut draw.ov;
            ov.overlay_draw_segments.clear();
            ov.overlay_order_items.clear();
            ov.overlay_shape_instances.clear();
            ov.overlay_text_verts.clear();
            ov.overlay_retained_draws.clear();
            ov.overlay_retained_shape_draws.clear();
            ov.overlay_instances_ready = false;
            // The viewport uniform carries the layer-to-clip map after the
            // fields the overlay prepare writes, so it is made at full size
            // here before the prepare would make a small one.
            if ov
                .overlay_viewport_buf
                .as_ref()
                .is_none_or(|b| b.size() < WORLD_VIEWPORT_BYTES)
            {
                ov.overlay_viewport_buf =
                    Some(device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some("overlay_world_viewport_uniform"),
                        size: WORLD_VIEWPORT_BYTES,
                        usage: crate::gpu::BufferUsages::UNIFORM
                            | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }
            ov.layer_plan.build(&layer.frame);
            ov.layer_plan.flatten();
            self.prepare_overlay_labels(device, queue, &space, ov);
            self.prepare_overlay_shapes(device, queue, &space, ov);
            if let Some(cb) = self.finalize_overlay_draw_order(device, queue, &space, ov) {
                sink.push(cb);
            }
            if ov.main_segments().is_empty() {
                continue;
            }
            let mut uniform = [0.0f32; 20];
            uniform[..4].copy_from_slice(&[space.size[0], space.size[1], view.ppp, 0.0]);
            uniform[4..].copy_from_slice(&view.layer_to_clip.to_cols_array());
            if let Some(buf) = &ov.overlay_viewport_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&uniform));
            }
            let key = WorldPassKey {
                depth_test: layer.depth_test,
                ..pass
            };
            for seg in ov.main_segments() {
                let pipe = if seg.is_shape() {
                    WorldPipe::Shape
                } else if seg.is_text() {
                    WorldPipe::Text
                } else {
                    continue;
                };
                self.resources
                    .ensure_overlay_world_pipeline(device, key, pipe, seg.blend);
            }
            worlds.sequence.push(i);
        }
        // Painter's order between layers: by `order`, then submission.
        let order_of = |i: &usize| worlds.layers[*i].order;
        let mut sequence = std::mem::take(&mut worlds.sequence);
        sequence.sort_by_key(order_of);
        worlds.sequence = sequence;
        worlds.pass = Some(pass);

        self.overlay_retained_counters = counters;
        self.overlay_tess_cache.tessellations = tessellations;
        self.viewport_slots[vp_idx].world_overlays = worlds;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An orthographic camera looking down -Z at the XY plane, `units` world
    /// units across a target `px` device pixels wide.
    fn ortho_view_proj(units_per_pixel: f32, px: [f32; 2]) -> glam::Mat4 {
        let hw = px[0] * 0.5 * units_per_pixel;
        let hh = px[1] * 0.5 * units_per_pixel;
        let proj = glam::Mat4::orthographic_rh(-hw, hw, -hh, hh, 0.1, 100.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0, -2.0, 10.0),
            glam::Vec3::new(3.0, -2.0, 0.0),
            glam::Vec3::Y,
        );
        proj * view
    }

    #[test]
    fn a_unit_covers_the_cameras_pixels_per_unit_at_any_zoom() {
        let px = [640.0, 360.0];
        let layer =
            crate::renderer::types::WorldOverlayLayer::new(crate::renderer::types::WorldPlane::Xy)
                .with_origin([1.0, 7.0])
                .with_elevation(0.25);
        for units_per_pixel in [0.01_f32, 0.05, 0.4] {
            let m = ortho_view_proj(units_per_pixel, px) * layer.layer_to_world(glam::DVec3::ZERO);
            let view = world_layer_view(m, px).unwrap();
            let expect = 1.0 / units_per_pixel;
            assert!(
                (view.ppp - expect).abs() / expect < 1.0e-4,
                "{units_per_pixel}: {} px per unit, expected {expect}",
                view.ppp
            );
            // A hairline is one device pixel: 1 / ppp layer units, which the
            // camera maps to one pixel on screen.
            let a = m * glam::Vec4::new(2.0, 3.0, 0.0, 1.0);
            let b = m * glam::Vec4::new(2.0, 3.0 + 1.0 / view.ppp, 0.0, 1.0);
            let dy_px = ((b.y / b.w) - (a.y / a.w)).abs() * px[1] * 0.5;
            assert!(
                (dy_px - 1.0).abs() < 1.0e-3,
                "{units_per_pixel}: {dy_px} px"
            );
        }
    }

    #[test]
    fn an_edge_on_layer_has_no_view() {
        let px = [640.0, 360.0];
        // The XZ plane seen from straight above is edge-on.
        let layer =
            crate::renderer::types::WorldOverlayLayer::new(crate::renderer::types::WorldPlane::Xz);
        let m = ortho_view_proj(0.05, px) * layer.layer_to_world(glam::DVec3::ZERO);
        assert!(world_layer_view(m, px).is_none());
    }
}
