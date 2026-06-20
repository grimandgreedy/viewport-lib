//! Overlay prepare passes: labels, rects, polylines, scalar bars, rulers,
//! loading bars, and SDF overlay shapes.

use super::*;

impl ViewportRenderer {
    pub(super) fn prepare_overlay_labels(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) {
        // ---------------------------------------------------------------
        // Overlay labels
        // ---------------------------------------------------------------
        // Rects and labels are merged into a single z_order-sorted vertex buffer so
        // that rects and labels at the same z_order interleave correctly (e.g. a
        // console background rect at i32::MAX sits below console text at i32::MAX
        // but above HUD labels at 0).
        self.label_gpu_data = None;
        self.overlay_rect_gpu_data = None;
        let has_overlay = !frame.overlays.labels.is_empty()
            || !frame.overlays.rects.is_empty()
            || !frame.overlays.polylines.is_empty();
        if has_overlay {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let view = &frame.camera.render_camera.view;
                let proj = &frame.camera.render_camera.projection;

                // Each item (label or rect) contributes a (z_order, verts) batch.
                // Rects are pushed before labels so that after a stable sort, items at
                // equal z_order always draw rect (background) before label (text).
                let mut batches: Vec<(i32, Vec<crate::resources::OverlayTextVertex>)> = Vec::new();

                // --- Rects (processed first so equal-z_order rects draw before labels) ---
                for rect in &frame.overlays.rects {
                    if rect.opacity <= 0.0 {
                        continue;
                    }
                    let x0 = rect.position[0];
                    let y0 = rect.position[1];
                    let x1 = x0 + rect.size[0];
                    let y1 = y0 + rect.size[1];

                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                    if rect.border_width > 0.0 {
                        let bw = rect.border_width;
                        let mut bc = rect.border_colour;
                        bc[3] *= rect.opacity;
                        emit_rounded_quad(
                            &mut batch,
                            x0 - bw,
                            y0 - bw,
                            x1 + bw,
                            y1 + bw,
                            rect.corner_radius + bw,
                            bc,
                            vp_w,
                            vp_h,
                        );
                    }

                    let mut fc = rect.colour;
                    fc[3] *= rect.opacity;
                    emit_rounded_quad(
                        &mut batch,
                        x0,
                        y0,
                        x1,
                        y1,
                        rect.corner_radius,
                        fc,
                        vp_w,
                        vp_h,
                    );

                    batches.push((rect.z_order, batch));
                }

                // --- Polylines (between rects and labels in z order) ---
                for poly in &frame.overlays.polylines {
                    if poly.points.len() < 2 || poly.opacity <= 0.0 {
                        continue;
                    }
                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                    if poly.closed && poly.texture.is_none() {
                        if let Some(fill) = &poly.fill {
                            emit_filled_polyline(
                                &mut batch,
                                &poly.points,
                                fill,
                                poly.opacity,
                                vp_w,
                                vp_h,
                            );
                        }
                    }
                    if poly.thickness > 0.0 {
                        let mut colour = poly.colour;
                        colour[3] *= poly.opacity;
                        batch.extend(tessellate_polyline(
                            &poly.points,
                            poly.thickness,
                            poly.closed,
                            poly.join,
                            poly.mitre_limit,
                            colour,
                            vp_w,
                            vp_h,
                        ));
                    }
                    if !batch.is_empty() {
                        batches.push((poly.z_order, batch));
                    }
                }

                // --- Labels ---
                for label in &frame.overlays.labels {
                    if label.text.is_empty() || label.opacity <= 0.0 {
                        continue;
                    }

                    let screen_pos = if let Some(sa) = label.screen_anchor {
                        Some(sa)
                    } else if let Some(wa) = label.world_anchor {
                        project_to_screen(wa, view, proj, vp_w, vp_h)
                    } else {
                        continue;
                    };
                    let Some(anchor_px) = screen_pos else {
                        continue;
                    };

                    let opacity = label.opacity.clamp(0.0, 1.0);

                    let layout = if let Some(max_w) = label.max_width {
                        self.resources.glyph_atlas.layout_text_wrapped(
                            &label.text,
                            label.font_size,
                            label.font,
                            max_w,
                            device,
                        )
                    } else {
                        self.resources.glyph_atlas.layout_text(
                            &label.text,
                            label.font_size,
                            label.font,
                            device,
                        )
                    };

                    let font_index = label.font.map_or(0, |h| h.0);
                    let ascent = self
                        .resources
                        .glyph_atlas
                        .font_ascent(font_index, label.font_size);

                    let align_offset = match label.anchor_align {
                        crate::renderer::types::LabelAnchor::Leading => 6.0,
                        crate::renderer::types::LabelAnchor::Center => -layout.total_width * 0.5,
                        crate::renderer::types::LabelAnchor::Trailing => -layout.total_width - 6.0,
                    };

                    let text_x = anchor_px[0] + align_offset + label.offset[0];
                    let text_y = anchor_px[1] - layout.height * 0.5 + label.offset[1];

                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                    if label.background {
                        let pad = label.padding;
                        let bx0 = text_x - pad;
                        let by0 = text_y - pad;
                        let bx1 = text_x + layout.total_width + pad;
                        let by1 = text_y + layout.height + pad;
                        let bg_colour = apply_opacity(label.background_colour, opacity);
                        if label.border_radius > 0.0 {
                            emit_rounded_quad(
                                &mut batch,
                                bx0,
                                by0,
                                bx1,
                                by1,
                                label.border_radius,
                                bg_colour,
                                vp_w,
                                vp_h,
                            );
                        } else {
                            emit_solid_quad(&mut batch, bx0, by0, bx1, by1, bg_colour, vp_w, vp_h);
                        }
                    }

                    if label.leader_line {
                        if let Some(wa) = label.world_anchor {
                            let world_px = project_to_screen(wa, view, proj, vp_w, vp_h);
                            if let Some(wp) = world_px {
                                emit_line_quad(
                                    &mut batch,
                                    wp[0],
                                    wp[1],
                                    text_x,
                                    text_y + layout.height * 0.5,
                                    1.5,
                                    apply_opacity(label.leader_colour, opacity),
                                    vp_w,
                                    vp_h,
                                );
                            }
                        }
                    }

                    let text_colour = apply_opacity(label.colour, opacity);
                    for gq in &layout.quads {
                        let gx = text_x + gq.pos[0];
                        let gy = text_y + ascent + gq.pos[1];
                        emit_textured_quad(
                            &mut batch,
                            gx,
                            gy,
                            gx + gq.size[0],
                            gy + gq.size[1],
                            gq.uv_min,
                            gq.uv_max,
                            text_colour,
                            vp_w,
                            vp_h,
                        );
                    }

                    batches.push((label.z_order, batch));
                }

                // Upload atlas if new glyphs were rasterized.
                self.resources.glyph_atlas.upload_if_dirty(queue);

                // Stable sort preserves rect-before-label order for equal z_order values.
                batches.sort_by_key(|(z, _)| *z);
                let verts: Vec<crate::resources::OverlayTextVertex> =
                    batches.into_iter().flat_map(|(_, v)| v).collect();

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("overlay_label_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_label_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.label_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }
    }

    pub(super) fn prepare_scalar_bars(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) {
        // ---------------------------------------------------------------
        // Scalar bars
        // ---------------------------------------------------------------
        self.scalar_bar_gpu_data = None;
        if !frame.overlays.scalar_bars.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                for bar in &frame.overlays.scalar_bars {
                    // Clone the LUT immediately so the immutable borrow on self.resources
                    // is released before the mutable glyph_atlas borrow below.
                    let Some(lut) = self
                        .resources
                        .get_colourmap_rgba(bar.colourmap_id)
                        .map(|l| l.to_vec())
                    else {
                        continue;
                    };

                    let is_vertical = matches!(
                        bar.orientation,
                        crate::renderer::types::ScalarBarOrientation::Vertical
                    );
                    let reversed = bar.ticks_reversed;

                    // Effective font sizes.
                    let tick_fs = bar.font_size;
                    let title_fs = bar.title_font_size.unwrap_or(bar.font_size);
                    let font_index = bar.font.map_or(0, |h| h.0);

                    // Actual pixel dimensions of the gradient strip.
                    let (strip_w, strip_h) = if is_vertical {
                        (bar.bar_width_px, bar.bar_length_px)
                    } else {
                        (bar.bar_length_px, bar.bar_width_px)
                    };

                    // Pre-compute tick texts and their widths so the background box
                    // can be sized to cover the tick labels.
                    let tick_count = bar.tick_count.max(2);
                    let mut tick_data: Vec<(String, f32, f32)> = Vec::new(); // (text, total_w, height)
                    let mut max_tick_w = 0.0f32;
                    let mut tick_h = 0.0f32;
                    for i in 0..tick_count {
                        let t = i as f32 / (tick_count - 1) as f32;
                        let value = bar.scalar_min + t * (bar.scalar_max - bar.scalar_min);
                        let text = format!("{value:.2}");
                        let layout = self
                            .resources
                            .glyph_atlas
                            .layout_text(&text, tick_fs, bar.font, device);
                        max_tick_w = max_tick_w.max(layout.total_width);
                        tick_h = layout.height;
                        tick_data.push((text, layout.total_width, layout.height));
                    }

                    // Vertical space reserved above the gradient strip.
                    // In vertical mode the top/bottom tick labels are centred on the strip
                    // endpoints, so they each overhang by tick_h/2. title_h must absorb the
                    // top overhang AND leave a gap so the title text does not touch the tick.
                    let half_tick = tick_h / 2.0;
                    let title_h = if bar.title.is_some() {
                        // title text height + small gap + top-tick overhang
                        title_fs + 4.0 + half_tick
                    } else {
                        // no title, but still need room for the top-tick overhang
                        half_tick
                    };

                    // Pre-compute title width before bar_x/bar_y so the overhang can
                    // be used to push the strip inward and prevent clipping.
                    let title_w = if let Some(ref t) = bar.title {
                        self.resources
                            .glyph_atlas
                            .layout_text(t, title_fs, bar.font, device)
                            .total_width
                    } else {
                        0.0
                    };

                    // How far title / tick labels spill beyond the strip on each side.
                    // Vertical: title centred on the narrow strip, ticks to the right.
                    //   left side: title overhang only.
                    //   right side: ticks dominate (strip_w + 4 + max_tick_w).
                    // Horizontal: both title and tick labels can overhang left/right equally.
                    let bg_pad = 4.0;
                    let (inset_left, inset_right) = if is_vertical {
                        let title_oh = ((title_w - strip_w) / 2.0).max(0.0);
                        let right_extent = 4.0 + max_tick_w + bg_pad; // relative to strip right edge
                        (title_oh + bg_pad, right_extent)
                    } else {
                        let title_oh = ((title_w - strip_w) / 2.0).max(0.0);
                        let tick_oh = max_tick_w / 2.0;
                        let side = title_oh.max(tick_oh) + bg_pad;
                        (side, side)
                    };

                    // How far content hangs below the strip bottom (used to keep the
                    // background box flush with margin_px on the bottom-anchored side).
                    // Vertical: bottom tick label is centred on the strip endpoint -> half_tick.
                    // Horizontal: tick labels sit fully below the strip -> 3 + tick_h.
                    let bottom_overhang = if is_vertical { half_tick } else { 3.0 + tick_h };

                    // Top-left of the gradient strip.
                    // bg_pad is added/subtracted here so that the background box edge lands
                    // exactly at margin_px from the viewport edge on the anchored side.
                    //   Top anchor:    bg_y0 = bar_y - title_h - bg_pad  =>  set bar_y = margin_px + title_h + bg_pad
                    //   Bottom anchor: bg_y1 = bar_y + strip_h + bottom_overhang + bg_pad  =>  bar_y = vp_h - margin_px - strip_h - bottom_overhang - bg_pad
                    let (bar_x, bar_y) = match bar.anchor {
                        crate::renderer::types::ScalarBarAnchor::TopLeft => {
                            (bar.margin_px + inset_left, bar.margin_px + title_h + bg_pad)
                        }
                        crate::renderer::types::ScalarBarAnchor::TopRight => (
                            vp_w - bar.margin_px - strip_w - inset_right,
                            bar.margin_px + title_h + bg_pad,
                        ),
                        crate::renderer::types::ScalarBarAnchor::BottomLeft => (
                            bar.margin_px + inset_left,
                            vp_h - bar.margin_px - strip_h - bottom_overhang - bg_pad,
                        ),
                        crate::renderer::types::ScalarBarAnchor::BottomRight => (
                            vp_w - bar.margin_px - strip_w - inset_right,
                            vp_h - bar.margin_px - strip_h - bottom_overhang - bg_pad,
                        ),
                    };

                    // Background box: now that bar_x/bar_y are inset, the box stays on screen.
                    let (bg_x0, bg_y0, bg_x1, bg_y1) = if is_vertical {
                        let title_right = bar_x + (strip_w + title_w) / 2.0;
                        let ticks_right = bar_x + strip_w + 4.0 + max_tick_w;
                        (
                            bar_x - bg_pad - ((title_w - strip_w) / 2.0).max(0.0),
                            bar_y - title_h - bg_pad,
                            ticks_right.max(title_right) + bg_pad,
                            bar_y + strip_h + half_tick + bg_pad,
                        )
                    } else {
                        let title_overhang = ((title_w - strip_w) / 2.0).max(0.0);
                        let tick_overhang = max_tick_w / 2.0;
                        let side_pad = title_overhang.max(tick_overhang);
                        let bottom = bar_y + strip_h + 3.0 + tick_h + bg_pad;
                        (
                            bar_x - bg_pad - side_pad,
                            bar_y - title_h - bg_pad,
                            bar_x + strip_w + bg_pad + side_pad,
                            bottom,
                        )
                    };
                    emit_rounded_quad(
                        &mut verts,
                        bg_x0,
                        bg_y0,
                        bg_x1,
                        bg_y1,
                        3.0,
                        bar.background_colour,
                        vp_w,
                        vp_h,
                    );

                    // Gradient strip: 64 solid quads sampled from the colourmap LUT.
                    let steps: usize = 64;
                    for s in 0..steps {
                        let (qx0, qy0, qx1, qy1, t) = if is_vertical {
                            // Default: top = max (t=1). Reversed: top = min (t=0).
                            let t = if reversed {
                                s as f32 / (steps - 1) as f32
                            } else {
                                1.0 - s as f32 / (steps - 1) as f32
                            };
                            let step_h = strip_h / steps as f32;
                            let sy = bar_y + s as f32 * step_h;
                            (bar_x, sy, bar_x + strip_w, sy + step_h + 0.5, t)
                        } else {
                            // Default: left = min (t=0). Reversed: left = max (t=1).
                            let t = if reversed {
                                1.0 - s as f32 / (steps - 1) as f32
                            } else {
                                s as f32 / (steps - 1) as f32
                            };
                            let step_w = strip_w / steps as f32;
                            let sx = bar_x + s as f32 * step_w;
                            (sx, bar_y, sx + step_w + 0.5, bar_y + strip_h, t)
                        };
                        let lut_idx = (t * 255.0).clamp(0.0, 255.0) as usize;
                        let [r, g, b, a] = lut[lut_idx];
                        let colour = [
                            r as f32 / 255.0,
                            g as f32 / 255.0,
                            b as f32 / 255.0,
                            a as f32 / 255.0,
                        ];
                        emit_solid_quad(&mut verts, qx0, qy0, qx1, qy1, colour, vp_w, vp_h);
                    }

                    // Tick labels.
                    let ascent = self.resources.glyph_atlas.font_ascent(font_index, tick_fs);
                    for (i, (text, tw, th)) in tick_data.iter().enumerate() {
                        let t = i as f32 / (tick_count - 1) as f32;
                        let layout = self
                            .resources
                            .glyph_atlas
                            .layout_text(text, tick_fs, bar.font, device);

                        let (lx, ly) = if is_vertical {
                            // Place text to the right of the strip, vertically centered
                            // on its tick position.
                            // Default: top=max -> progress = 1.0-t puts max at top.
                            // Reversed: top=min -> progress = t puts min at top.
                            let progress = if reversed { t } else { 1.0 - t };
                            let tick_y = bar_y + progress * strip_h;
                            (bar_x + strip_w + 4.0, tick_y - th * 0.5)
                        } else {
                            // Place text below the strip, horizontally centered on its tick.
                            // Default: left=min -> tick at t*strip_w.
                            // Reversed: left=max -> tick at (1-t)*strip_w.
                            let frac = if reversed { 1.0 - t } else { t };
                            let tick_x = bar_x + frac * strip_w;
                            (tick_x - tw * 0.5, bar_y + strip_h + 3.0)
                        };
                        let _ = (tw, th); // used above

                        for gq in &layout.quads {
                            let gx = lx + gq.pos[0];
                            let gy = ly + ascent + gq.pos[1];
                            emit_textured_quad(
                                &mut verts,
                                gx,
                                gy,
                                gx + gq.size[0],
                                gy + gq.size[1],
                                gq.uv_min,
                                gq.uv_max,
                                bar.label_colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }

                    // Optional title above the gradient strip.
                    if let Some(ref title_text) = bar.title {
                        let layout = self
                            .resources
                            .glyph_atlas
                            .layout_text(title_text, title_fs, bar.font, device);
                        let title_ascent =
                            self.resources.glyph_atlas.font_ascent(font_index, title_fs);
                        // Centre the title over the gradient strip.
                        let tx = bar_x + (strip_w - layout.total_width) * 0.5;
                        let ty = bar_y - title_h;
                        for gq in &layout.quads {
                            let gx = tx + gq.pos[0];
                            let gy = ty + title_ascent + gq.pos[1];
                            emit_textured_quad(
                                &mut verts,
                                gx,
                                gy,
                                gx + gq.size[0],
                                gy + gq.size[1],
                                gq.uv_min,
                                gq.uv_max,
                                bar.label_colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }
                }

                // Upload any newly rasterized glyphs (may overlap with label upload above).
                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("overlay_scalar_bar_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_scalar_bar_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.scalar_bar_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }
    }

    pub(super) fn prepare_rulers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) {
        // ---------------------------------------------------------------
        // Rulers
        // ---------------------------------------------------------------
        self.ruler_gpu_data = None;
        if !frame.overlays.rulers.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let view = &frame.camera.render_camera.view;
                let proj = &frame.camera.render_camera.projection;

                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                for ruler in &frame.overlays.rulers {
                    // Project both endpoints to NDC (returns None only if behind camera).
                    let start_ndc = project_to_ndc(ruler.start, view, proj);
                    let end_ndc = project_to_ndc(ruler.end, view, proj);

                    // Cull entirely when either endpoint is behind the camera.
                    let (Some(sndc), Some(endc)) = (start_ndc, end_ndc) else {
                        continue;
                    };

                    // Clip the segment to the viewport NDC box [-1,1]^2.
                    // This keeps the line visible when only one end is off-screen sideways.
                    let Some((csndc, cendc)) = clip_line_ndc(sndc, endc) else {
                        continue;
                    };

                    let [sx, sy] = ndc_to_screen_px(csndc, vp_w, vp_h);
                    let [ex, ey] = ndc_to_screen_px(cendc, vp_w, vp_h);

                    // Track which original endpoints are within the viewport (for end caps).
                    let start_on_screen = ndc_in_viewport(sndc);
                    let end_on_screen = ndc_in_viewport(endc);

                    // Main ruler line.
                    emit_line_quad(
                        &mut verts,
                        sx,
                        sy,
                        ex,
                        ey,
                        ruler.line_width_px,
                        ruler.colour,
                        vp_w,
                        vp_h,
                    );

                    // End caps only at endpoints that are actually on screen.
                    if ruler.end_caps {
                        let dx = ex - sx;
                        let dy = ey - sy;
                        let len = (dx * dx + dy * dy).sqrt().max(0.001);
                        let cap_half = 5.0;
                        let px = -dy / len * cap_half;
                        let py = dx / len * cap_half;

                        if start_on_screen {
                            emit_line_quad(
                                &mut verts,
                                sx - px,
                                sy - py,
                                sx + px,
                                sy + py,
                                ruler.line_width_px,
                                ruler.colour,
                                vp_w,
                                vp_h,
                            );
                        }
                        if end_on_screen {
                            emit_line_quad(
                                &mut verts,
                                ex - px,
                                ey - py,
                                ex + px,
                                ey + py,
                                ruler.line_width_px,
                                ruler.colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }

                    // Distance label: always shows true 3D distance.
                    // Place it at the midpoint of the visible (clipped) segment.
                    let start_world = glam::Vec3::from(ruler.start);
                    let end_world = glam::Vec3::from(ruler.end);
                    let distance = (end_world - start_world).length();
                    let text = format_ruler_distance(distance, ruler.label_format.as_deref());

                    let mid_x = (sx + ex) * 0.5;
                    let mid_y = (sy + ey) * 0.5;

                    let layout = self.resources.glyph_atlas.layout_text(
                        &text,
                        ruler.font_size,
                        ruler.font,
                        device,
                    );
                    let font_index = ruler.font.map_or(0, |h| h.0);
                    let ascent = self
                        .resources
                        .glyph_atlas
                        .font_ascent(font_index, ruler.font_size);

                    // Center the label above the midpoint with a small gap.
                    let lx = mid_x - layout.total_width * 0.5;
                    let ly = mid_y - layout.height - 6.0;

                    // Semi-transparent background box.
                    let pad = 3.0;
                    emit_solid_quad(
                        &mut verts,
                        lx - pad,
                        ly - pad,
                        lx + layout.total_width + pad,
                        ly + layout.height + pad,
                        [0.0, 0.0, 0.0, 0.55],
                        vp_w,
                        vp_h,
                    );

                    // Glyph quads.
                    for gq in &layout.quads {
                        let gx = lx + gq.pos[0];
                        let gy = ly + ascent + gq.pos[1];
                        emit_textured_quad(
                            &mut verts,
                            gx,
                            gy,
                            gx + gq.size[0],
                            gy + gq.size[1],
                            gq.uv_min,
                            gq.uv_max,
                            ruler.label_colour,
                            vp_w,
                            vp_h,
                        );
                    }
                }

                // Upload any newly rasterized glyphs.
                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("overlay_ruler_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("overlay_ruler_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.ruler_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }
    }

    pub(super) fn prepare_loading_bars(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) {
        // ---------------------------------------------------------------
        // Loading bars
        // ---------------------------------------------------------------
        self.loading_bar_gpu_data = None;
        if !frame.overlays.loading_bars.is_empty() {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                for bar in &frame.overlays.loading_bars {
                    // Bar top-left corner based on anchor.
                    let bar_x = vp_w * 0.5 - bar.width_px * 0.5;
                    let bar_y = match bar.anchor {
                        crate::renderer::types::LoadingBarAnchor::TopCenter => bar.margin_px,
                        crate::renderer::types::LoadingBarAnchor::Center => {
                            vp_h * 0.5 - bar.height_px * 0.5
                        }
                        crate::renderer::types::LoadingBarAnchor::BottomCenter => {
                            vp_h - bar.margin_px - bar.height_px
                        }
                    };

                    // Label above (TopCenter: below) the bar.
                    if let Some(ref text) = bar.label {
                        let layout = self.resources.glyph_atlas.layout_text(
                            text,
                            bar.font_size,
                            bar.font,
                            device,
                        );
                        let font_index = bar.font.map_or(0, |h| h.0);
                        let ascent = self
                            .resources
                            .glyph_atlas
                            .font_ascent(font_index, bar.font_size);
                        let label_gap = 5.0;
                        let lx = bar_x + bar.width_px * 0.5 - layout.total_width * 0.5;
                        let ly = match bar.anchor {
                            crate::renderer::types::LoadingBarAnchor::TopCenter => {
                                bar_y + bar.height_px + label_gap
                            }
                            _ => bar_y - layout.height - label_gap,
                        };
                        for gq in &layout.quads {
                            let gx = lx + gq.pos[0];
                            let gy = ly + ascent + gq.pos[1];
                            emit_textured_quad(
                                &mut verts,
                                gx,
                                gy,
                                gx + gq.size[0],
                                gy + gq.size[1],
                                gq.uv_min,
                                gq.uv_max,
                                bar.label_colour,
                                vp_w,
                                vp_h,
                            );
                        }
                    }

                    // Background rectangle.
                    emit_rounded_quad(
                        &mut verts,
                        bar_x,
                        bar_y,
                        bar_x + bar.width_px,
                        bar_y + bar.height_px,
                        bar.corner_radius,
                        bar.background_colour,
                        vp_w,
                        vp_h,
                    );

                    // Fill rectangle clipped to progress fraction.
                    let fill_w = bar.width_px * bar.progress.clamp(0.0, 1.0);
                    if fill_w > 0.5 {
                        emit_rounded_quad(
                            &mut verts,
                            bar_x,
                            bar_y,
                            bar_x + fill_w,
                            bar_y + bar.height_px,
                            bar.corner_radius,
                            bar.fill_colour,
                            vp_w,
                            vp_h,
                        );
                    }
                }

                self.resources.glyph_atlas.upload_if_dirty(queue);

                if !verts.is_empty() {
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("loading_bar_vbuf"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let bgl = self.resources.overlay_text_bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text_sampler.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("loading_bar_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.resources.glyph_atlas.view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    self.loading_bar_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                    });
                }
            }
        }
    }

    pub(super) fn prepare_overlay_shapes(
        &mut self,
        device: &wgpu::Device,
        frame: &FrameData,
    ) {
        // ------------------------------------------------------------------
        // SDF overlay shapes
        // ------------------------------------------------------------------
        self.overlay_shape_gpu_data = None;
        let has_textured_polyline_fill = frame
            .overlays
            .polylines
            .iter()
            .any(|p| p.closed && p.texture.is_some() && p.opacity > 0.0 && p.points.len() >= 3);
        if !frame.overlays.shapes.is_empty() || has_textured_polyline_fill {
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                let mut sorted: Vec<&crate::renderer::types::OverlayShapeItem> =
                    frame.overlays.shapes.iter().collect();
                sorted.sort_by_key(|s| s.z_order);

                let has_solid = sorted
                    .iter()
                    .any(|s| s.texture.is_none() && s.backdrop_blur <= 0.0);
                let has_tex =
                    sorted.iter().any(|s| s.texture.is_some()) || has_textured_polyline_fill;
                let has_blur = sorted
                    .iter()
                    .any(|s| s.backdrop_blur > 0.0 && s.texture.is_none());
                if has_solid {
                    self.resources.ensure_overlay_shape_pipeline(device);
                }
                if has_tex || has_blur {
                    self.resources.ensure_overlay_shape_tex_pipeline(device);
                }
                if has_blur {
                    self.resources.ensure_backdrop_blur_pipeline(device);
                    self.resources.ensure_dyn_res_pipeline(device);
                }

                let mut solid_verts: Vec<crate::resources::OverlayShapeVertex> = Vec::new();
                // One vertex list per unique texture ID, in order of first appearance.
                let mut tex_groups: Vec<(u64, Vec<crate::resources::OverlayShapeTexVertex>)> =
                    Vec::new();
                // Blur backdrop vertices (share the tex vertex layout with screen UVs).
                let mut blur_verts: Vec<crate::resources::OverlayShapeTexVertex> = Vec::new();
                let mut max_blur_radius: f32 = 0.0;

                let overlay_time = frame.overlays.time;

                // First pass: collect clip-mask bounding rects keyed by
                // clip_mask_id. Shape positions are logical pixels; the
                // fragment shader's `clip_position.xy` reports framebuffer
                // (physical) pixels, so the rect is scaled by ppp here.
                let ppp = frame.camera.pixels_per_point;
                let mut clip_rects: std::collections::HashMap<u32, [f32; 4]> =
                    std::collections::HashMap::new();
                for shape in &sorted {
                    if let Some(id) = shape.clip_mask_id {
                        let x0 = shape.position[0] * ppp;
                        let y0 = shape.position[1] * ppp;
                        let x1 = (shape.position[0] + shape.size[0]) * ppp;
                        let y1 = (shape.position[1] + shape.size[1]) * ppp;
                        clip_rects.insert(id, [x0, y0, x1, y1]);
                    }
                }

                for shape_orig in &sorted {
                    // Mask-only shapes contribute a clip rectangle but are
                    // not drawn themselves.
                    if shape_orig.clip_mask_id.is_some() {
                        continue;
                    }
                    // Clone so per-frame animation overrides are local and
                    // the input frame data stays untouched.
                    let mut owned: crate::renderer::types::OverlayShapeItem = (*shape_orig).clone();
                    if let Some(track) = owned.animations.opacity {
                        // Multi-channel opacity track takes precedence over
                        // the legacy `animation` field.
                        owned.opacity = track.sample(overlay_time);
                        owned.animation = crate::renderer::types::OverlayAnimation::None;
                    }
                    if let Some(track) = owned.animations.position {
                        owned.position = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.size {
                        owned.size = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.fill {
                        if let crate::renderer::types::OverlayFill::Solid(_) = owned.fill {
                            owned.fill = crate::renderer::types::OverlayFill::Solid(
                                track.sample(overlay_time),
                            );
                        }
                    }
                    if let Some(track) = owned.animations.border {
                        owned.border_colour = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.rotation {
                        owned.rotation = track.sample(overlay_time);
                    }
                    // Path tracks override the matching linear track when
                    // both are set. `opacity_path` also takes precedence
                    // over the legacy `animation` field.
                    if let Some(track) = owned.animations.opacity_path.clone() {
                        owned.opacity = track.sample(overlay_time);
                        owned.animation = crate::renderer::types::OverlayAnimation::None;
                    }
                    if let Some(track) = owned.animations.position_path.clone() {
                        owned.position = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.size_path.clone() {
                        owned.size = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.fill_path.clone() {
                        if let crate::renderer::types::OverlayFill::Solid(_) = owned.fill {
                            owned.fill = crate::renderer::types::OverlayFill::Solid(
                                track.sample(overlay_time),
                            );
                        }
                    }
                    if let Some(track) = owned.animations.border_path.clone() {
                        owned.border_colour = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.rotation_path.clone() {
                        owned.rotation = track.sample(overlay_time);
                    }
                    let shape = &owned;
                    // Resolve animation to final opacity.
                    let resolved_opacity = match shape.animation {
                        crate::renderer::types::OverlayAnimation::None => shape.opacity,
                        crate::renderer::types::OverlayAnimation::FadeIn {
                            start_time,
                            duration,
                        } => {
                            let t = ((overlay_time - start_time) as f32 / duration.max(1e-6))
                                .clamp(0.0, 1.0);
                            shape.opacity * t
                        }
                        crate::renderer::types::OverlayAnimation::FadeOut {
                            start_time,
                            duration,
                        } => {
                            let t = ((overlay_time - start_time) as f32 / duration.max(1e-6))
                                .clamp(0.0, 1.0);
                            shape.opacity * (1.0 - t)
                        }
                        crate::renderer::types::OverlayAnimation::Pulse { start_time, period } => {
                            let t = ((overlay_time - start_time) as f32) / period.max(1e-6);
                            let wave = (t * std::f32::consts::TAU).sin() * 0.5 + 0.5;
                            shape.opacity * wave
                        }
                    };

                    if resolved_opacity <= 0.0 {
                        continue;
                    }

                    let hw = shape.size[0] * 0.5;
                    let hh = shape.size[1] * 0.5;
                    let cx = shape.position[0] + hw;
                    let cy = shape.position[1] + hh;

                    // Expand the bounding quad by border width, shadow extent,
                    // and AA so nothing gets clipped at the edges.
                    let shadow_pad = if shape.shadow_radius > 0.0 {
                        shape.shadow_radius
                            + shape.shadow_offset[0]
                                .abs()
                                .max(shape.shadow_offset[1].abs())
                    } else {
                        0.0
                    };
                    let pad = shape.border_width + shadow_pad + 1.0; // +1 for AA

                    // Extra quad expansion for shapes whose stroke extends
                    // beyond the item's position/size bounding box.
                    let extra_expand = match shape.shape {
                        crate::renderer::types::OverlayShape::Line { thickness, .. } => {
                            thickness * 0.5
                        }
                        _ => 0.0,
                    };
                    let ex = hw + pad + extra_expand;
                    let ey = hh + pad + extra_expand;

                    // Encode shape type and radii.
                    let (shape_type, radii) = match shape.shape {
                        crate::renderer::types::OverlayShape::Rect { corner_radius } => {
                            let r = corner_radius.min(hw).min(hh).max(0.0);
                            (0.0, [r, r, r, r])
                        }
                        crate::renderer::types::OverlayShape::RoundedRect { radii: r } => {
                            // Input: [tl, tr, br, bl].
                            // Shader expects iq convention: [tr, br, bl, tl].
                            let clamped = [
                                r[1].min(hw).min(hh).max(0.0), // top-right
                                r[2].min(hw).min(hh).max(0.0), // bottom-right
                                r[3].min(hw).min(hh).max(0.0), // bottom-left
                                r[0].min(hw).min(hh).max(0.0), // top-left
                            ];
                            (0.0, clamped)
                        }
                        crate::renderer::types::OverlayShape::Circle => (1.0, [0.0; 4]),
                        crate::renderer::types::OverlayShape::Ellipse => (2.0, [0.0; 4]),
                        crate::renderer::types::OverlayShape::Capsule => (3.0, [0.0; 4]),
                        crate::renderer::types::OverlayShape::Ring { inner_radius_frac } => {
                            (4.0, [inner_radius_frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Arc {
                            inner_radius_frac,
                            start_angle,
                            end_angle,
                        } => (
                            5.0,
                            [
                                inner_radius_frac.clamp(0.0, 1.0),
                                start_angle,
                                end_angle,
                                0.0,
                            ],
                        ),
                        crate::renderer::types::OverlayShape::Triangle { direction } => {
                            let dir_f = match direction {
                                crate::renderer::types::TriangleDirection::Up => 0.0,
                                crate::renderer::types::TriangleDirection::Down => 1.0,
                                crate::renderer::types::TriangleDirection::Left => 2.0,
                                crate::renderer::types::TriangleDirection::Right => 3.0,
                            };
                            (6.0, [dir_f, 0.0, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Line { thickness, cap } => {
                            let cap_f = match cap {
                                crate::renderer::types::LineCap::Round => 0.0,
                                crate::renderer::types::LineCap::Square => 1.0,
                            };
                            (7.0, [thickness * 0.5, cap_f, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Star {
                            points,
                            inner_radius_frac,
                        } => {
                            let n = points.max(3) as f32;
                            (8.0, [n, inner_radius_frac.clamp(0.0, 1.0), 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::RegularPolygon { sides } => {
                            let n = sides.max(3) as f32;
                            (9.0, [n, 0.0, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Cross { arm_width_frac } => {
                            (10.0, [arm_width_frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0])
                        }
                    };

                    // Resolve the fill into four colour stops + positions +
                    // a gradient type. Solid and 2-stop variants set
                    // count = 2 and use only stops[0..2]; multi-stop fills
                    // pack up to OVERLAY_MAX_GRADIENT_STOPS stops.
                    let mut stop_colours = [[0.0f32; 4]; 4];
                    let mut stop_positions = [0.0_f32, 1.0, 1.0, 1.0];
                    let stop_count: f32;
                    let gradient_params = match &shape.fill {
                        crate::renderer::types::OverlayFill::Solid(c) => {
                            stop_colours[0] = *c;
                            stop_colours[1] = *c;
                            stop_count = 0.0;
                            [0.0_f32, 0.0]
                        }
                        crate::renderer::types::OverlayFill::LinearGradient {
                            start_colour,
                            end_colour,
                            angle,
                        } => {
                            stop_colours[0] = *start_colour;
                            stop_colours[1] = *end_colour;
                            stop_count = 2.0;
                            [1.0_f32, *angle]
                        }
                        crate::renderer::types::OverlayFill::RadialGradient {
                            centre_colour,
                            edge_colour,
                        } => {
                            stop_colours[0] = *centre_colour;
                            stop_colours[1] = *edge_colour;
                            stop_count = 2.0;
                            [2.0_f32, 0.0]
                        }
                        crate::renderer::types::OverlayFill::ConicalGradient {
                            start_colour,
                            end_colour,
                            offset_angle,
                        } => {
                            stop_colours[0] = *start_colour;
                            stop_colours[1] = *end_colour;
                            stop_count = 2.0;
                            [3.0_f32, *offset_angle]
                        }
                        crate::renderer::types::OverlayFill::LinearGradientMulti {
                            stops,
                            angle,
                        } => {
                            stop_count = pack_stops(stops, &mut stop_colours, &mut stop_positions);
                            [1.0_f32, *angle]
                        }
                        crate::renderer::types::OverlayFill::RadialGradientMulti { stops } => {
                            stop_count = pack_stops(stops, &mut stop_colours, &mut stop_positions);
                            [2.0_f32, 0.0]
                        }
                        crate::renderer::types::OverlayFill::ConicalGradientMulti {
                            stops,
                            offset_angle,
                        } => {
                            stop_count = pack_stops(stops, &mut stop_colours, &mut stop_positions);
                            [3.0_f32, *offset_angle]
                        }
                    };
                    let start_colour = stop_colours[0];
                    let end_colour = stop_colours[1];
                    // Apply opacity to every stop (stops[0..4]) so multi-stop
                    // gradients fade as a whole when the item's opacity changes.
                    for colour in &mut stop_colours {
                        colour[3] *= resolved_opacity;
                    }
                    let fc = stop_colours[0];
                    let fc2 = stop_colours[1];
                    let _ = (start_colour, end_colour);
                    let mut bc = shape.border_colour;
                    bc[3] *= resolved_opacity;

                    let half_size = [hw, hh];

                    let mut sc = shape.shadow_colour;
                    sc[3] *= resolved_opacity;
                    let border_mode_f = match shape.border_mode {
                        crate::renderer::types::BorderMode::Inset => 0.0,
                        crate::renderer::types::BorderMode::Outer => 1.0,
                        crate::renderer::types::BorderMode::Center => 2.0,
                    };
                    // Pack the inset-shadow flag alongside border_mode in
                    // shadow_params.w. The shader decodes via (combined % 3)
                    // for border_mode and (combined >= 3) for inset.
                    let inset_flag = if shape.shadow_inset { 3.0 } else { 0.0 };
                    let shadow_params = [
                        shape.shadow_radius,
                        shape.shadow_offset[0],
                        shape.shadow_offset[1],
                        border_mode_f + inset_flag,
                    ];

                    // Emit 6 vertices (two triangles) for the bounding quad.
                    let corners_px = [
                        (cx - ex, cy - ey, -ex, -ey),
                        (cx + ex, cy - ey, ex, -ey),
                        (cx + ex, cy + ey, ex, ey),
                        (cx - ex, cy - ey, -ex, -ey),
                        (cx + ex, cy + ey, ex, ey),
                        (cx - ex, cy + ey, -ex, ey),
                    ];

                    if let Some(crate::renderer::types::OverlayTextureId(tid)) = shape.texture {
                        // Find or create a group for this texture ID.
                        let group_idx = tex_groups
                            .iter()
                            .position(|(id, _)| *id == tid)
                            .unwrap_or_else(|| {
                                tex_groups.push((tid, Vec::new()));
                                tex_groups.len() - 1
                            });
                        let group_verts = &mut tex_groups[group_idx].1;

                        // UV maps local_pos to [0,1] over the shape content area.
                        // hw/hh safe-guarded so we never divide by zero.
                        let hw_s = if hw > 0.0 { hw } else { 1.0 };
                        let hh_s = if hh > 0.0 { hh } else { 1.0 };

                        // 9-slice: convert pixel insets to texture UV ratios
                        // (using the bound texture's size) and to shape-fraction
                        // ratios for the shader's piecewise UV remap.
                        let (nine_uv, nine_frac, nine_extras_yzw) =
                            if let Some(ns) = shape.nine_slice {
                                let tex_size = self
                                    .resources
                                    .overlay_textures
                                    .get(tid as usize)
                                    .map(|t| t._texture.size())
                                    .map(|s| (s.width as f32, s.height as f32))
                                    .unwrap_or((1.0, 1.0));
                                let tw = tex_size.0.max(1.0);
                                let th = tex_size.1.max(1.0);
                                let shape_w = (shape.size[0]).max(1.0);
                                let shape_h = (shape.size[1]).max(1.0);
                                let inset_uv = [
                                    (ns.insets_px[0] / th).clamp(0.0, 0.5),
                                    (ns.insets_px[1] / tw).clamp(0.0, 0.5),
                                    (ns.insets_px[2] / th).clamp(0.0, 0.5),
                                    (ns.insets_px[3] / tw).clamp(0.0, 0.5),
                                ];
                                let inset_frac = [
                                    (ns.insets_px[0] / shape_h).clamp(0.0, 0.5),
                                    (ns.insets_px[1] / shape_w).clamp(0.0, 0.5),
                                    (ns.insets_px[2] / shape_h).clamp(0.0, 0.5),
                                    (ns.insets_px[3] / shape_w).clamp(0.0, 0.5),
                                ];
                                let centre = tile_mode_to_f(ns.centre_mode);
                                let edge = tile_mode_to_f(ns.edge_mode);
                                (inset_uv, inset_frac, [centre, edge, 1.0])
                            } else {
                                ([0.0; 4], [0.0; 4], [0.0, 0.0, 0.0])
                            };

                        let tt = shape.texture_transform;
                        let tt_a = [tt.offset[0], tt.offset[1], tt.scale[0], tt.scale[1]];
                        let tt_b = [
                            tt.rotation,
                            tile_mode_to_f(tt.tile_mode),
                            if tt.flip_x { 1.0 } else { 0.0 },
                            if tt.flip_y { 1.0 } else { 0.0 },
                        ];
                        for (px, py, lx, ly) in corners_px {
                            group_verts.push(crate::resources::OverlayShapeTexVertex {
                                position: px_to_ndc(px, py, vp_w, vp_h),
                                local_pos: [lx, ly],
                                fill_colour: fc,
                                border_colour: bc,
                                half_size,
                                radii,
                                border_width: shape.border_width,
                                shape_type,
                                uv: [(lx + hw_s) / (2.0 * hw_s), (ly + hh_s) / (2.0 * hh_s)],
                                shadow_colour: sc,
                                shadow_params,
                                extras: [
                                    0.0,
                                    nine_extras_yzw[0],
                                    nine_extras_yzw[1],
                                    nine_extras_yzw[2],
                                ],
                                nine_slice_uv: nine_uv,
                                nine_slice_frac: nine_frac,
                                texture_transform_a: tt_a,
                                texture_transform_b: tt_b,
                            });
                        }
                    } else if shape.backdrop_blur > 0.0 {
                        max_blur_radius = max_blur_radius.max(shape.backdrop_blur);
                        // Blur backdrop: same tex vertex layout but UV is screen-space.
                        for (px, py, lx, ly) in corners_px {
                            blur_verts.push(crate::resources::OverlayShapeTexVertex {
                                position: px_to_ndc(px, py, vp_w, vp_h),
                                local_pos: [lx, ly],
                                fill_colour: fc,
                                border_colour: bc,
                                half_size,
                                radii,
                                border_width: shape.border_width,
                                shape_type,
                                uv: [px / vp_w, py / vp_h],
                                shadow_colour: sc,
                                shadow_params,
                                extras: [1.0, 0.0, 0.0, 0.0],
                                nine_slice_uv: [0.0; 4],
                                nine_slice_frac: [0.0; 4],
                                texture_transform_a: [0.0, 0.0, 1.0, 1.0],
                                texture_transform_b: [0.0, 0.0, 0.0, 0.0],
                            });
                        }
                    } else {
                        // Look up the clip rect for this shape (if it has a
                        // clip_id). Missing or unmatched ids fall through to
                        // an all-zero rect, which the shader treats as no
                        // clipping.
                        let clip_rect = shape
                            .clip_id
                            .and_then(|id| clip_rects.get(&id).copied())
                            .unwrap_or([0.0, 0.0, 0.0, 0.0]);
                        // gradient_params is now vec4: [type, angle, stop_count, _pad]
                        let gp4 = [gradient_params[0], gradient_params[1], stop_count, 0.0];
                        for (px, py, lx, ly) in corners_px {
                            solid_verts.push(crate::resources::OverlayShapeVertex {
                                position: px_to_ndc(px, py, vp_w, vp_h),
                                local_pos: [lx, ly],
                                fill_colour: fc,
                                border_colour: bc,
                                half_size,
                                radii,
                                border_width: shape.border_width,
                                shape_type,
                                fill_colour2: fc2,
                                gradient_params: gp4,
                                shadow_colour: sc,
                                shadow_params,
                                clip_rect,
                                rotation: shape.rotation,
                                stop_colour_c: stop_colours[2],
                                stop_colour_d: stop_colours[3],
                                stop_positions,
                            });
                        }
                    }
                }

                for poly in &frame.overlays.polylines {
                    let Some(crate::renderer::types::OverlayTextureId(tid)) = poly.texture else {
                        continue;
                    };
                    if !poly.closed || poly.opacity <= 0.0 || poly.points.len() < 3 {
                        continue;
                    }
                    let Some((min, max)) = polyline_bounds(&poly.points) else {
                        continue;
                    };
                    let tris = triangulate_polygon(&poly.points);
                    if tris.is_empty() {
                        continue;
                    }
                    let group_idx = tex_groups
                        .iter()
                        .position(|(id, _)| *id == tid)
                        .unwrap_or_else(|| {
                            tex_groups.push((tid, Vec::new()));
                            tex_groups.len() - 1
                        });
                    let group_verts = &mut tex_groups[group_idx].1;
                    let size = [(max[0] - min[0]).max(1e-6), (max[1] - min[1]).max(1e-6)];
                    let centre = [min[0] + size[0] * 0.5, min[1] + size[1] * 0.5];
                    let half_size = [size[0] * 0.5, size[1] * 0.5];
                    let mut tint = match &poly.fill {
                        Some(crate::renderer::types::OverlayFill::Solid(c)) => *c,
                        _ => [1.0, 1.0, 1.0, 1.0],
                    };
                    tint[3] *= poly.opacity;
                    let explicit_uvs = poly
                        .uvs
                        .as_ref()
                        .filter(|uvs| uvs.len() == poly.points.len());
                    let tt = poly.texture_transform;
                    let tt_a = [tt.offset[0], tt.offset[1], tt.scale[0], tt.scale[1]];
                    let tt_b = [
                        tt.rotation,
                        tile_mode_to_f(tt.tile_mode),
                        if tt.flip_x { 1.0 } else { 0.0 },
                        if tt.flip_y { 1.0 } else { 0.0 },
                    ];
                    for tri in tris {
                        for idx in tri {
                            let p = poly.points[idx];
                            let local = [p[0] - centre[0], p[1] - centre[1]];
                            let uv = explicit_uvs.map_or(
                                [(p[0] - min[0]) / size[0], (p[1] - min[1]) / size[1]],
                                |uvs| uvs[idx],
                            );
                            group_verts.push(crate::resources::OverlayShapeTexVertex {
                                position: px_to_ndc(p[0], p[1], vp_w, vp_h),
                                local_pos: local,
                                fill_colour: tint,
                                border_colour: [0.0; 4],
                                half_size,
                                radii: [0.0; 4],
                                border_width: 0.0,
                                shape_type: 0.0,
                                uv,
                                shadow_colour: [0.0; 4],
                                shadow_params: [0.0; 4],
                                extras: [0.0; 4],
                                nine_slice_uv: [0.0; 4],
                                nine_slice_frac: [0.0; 4],
                                texture_transform_a: tt_a,
                                texture_transform_b: tt_b,
                            });
                        }
                    }
                }

                let solid_vbuf = if !solid_verts.is_empty() {
                    Some(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("overlay_shape_vbuf"),
                            contents: bytemuck::cast_slice(&solid_verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                    )
                } else {
                    None
                };

                let mut tex_batches = Vec::new();
                if has_tex {
                    if let (Some(bgl), Some(sampler)) = (
                        self.resources.overlay_shape_tex_bgl.as_ref(),
                        self.resources.overlay_shape_tex_sampler.as_ref(),
                    ) {
                        for (tid, verts) in &tex_groups {
                            if verts.is_empty() {
                                continue;
                            }
                            let tidx = *tid as usize;
                            if tidx >= self.resources.overlay_textures.len() {
                                continue;
                            }
                            let view = &self.resources.overlay_textures[tidx].view;
                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("overlay_shape_tex_bg"),
                                layout: bgl,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(sampler),
                                    },
                                ],
                            });
                            let vertex_buf =
                                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("overlay_shape_tex_vbuf"),
                                    contents: bytemuck::cast_slice(verts),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });
                            tex_batches.push(crate::resources::OverlayShapeTexBatch {
                                vertex_buf,
                                vertex_count: verts.len() as u32,
                                bind_group,
                            });
                        }
                    }
                }

                let blur_vbuf = if !blur_verts.is_empty() {
                    Some(
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("overlay_shape_blur_vbuf"),
                            contents: bytemuck::cast_slice(&blur_verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                    )
                } else {
                    None
                };

                if solid_vbuf.is_some() || !tex_batches.is_empty() || blur_vbuf.is_some() {
                    self.overlay_shape_gpu_data = Some(crate::resources::OverlayShapeGpuData {
                        vertex_buf: solid_vbuf,
                        vertex_count: solid_verts.len() as u32,
                        tex_batches,
                        blur_vertex_buf: blur_vbuf,
                        blur_vertex_count: blur_verts.len() as u32,
                        max_blur_radius,
                    });
                }
            }
        }
    }

}
