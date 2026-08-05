//! Off-screen scene capture for baked-lighting generation.
//!
//! [`ViewportRenderer::capture_hdr`] renders the lit scene from a caller-chosen
//! camera into the renderer's pre-tonemap `Rgba16Float` scene target and reads
//! the linear radiance back to the CPU. Unlike
//! [`render_offscreen`](ViewportRenderer::render_offscreen), which returns
//! tone-mapped LDR bytes, the values here keep their full HDR range, so a bright
//! sky or direct sun stays above 1.0. This is the shared input for baking SH
//! light probes and reflection probes, and for a reference lightmap solve.
//!
//! No `wgpu::Surface` is required: the capture drives the same headless render
//! path as `render_offscreen`, so it runs from a bake with no presentable frame.

use super::*;

/// Linear HDR radiance read back from a capture: row-major `width * height`
/// RGBA texels, four `f32` channels each.
#[derive(Clone, Debug)]
pub struct CapturedHdr {
    /// Capture width in pixels.
    pub width: u32,
    /// Capture height in pixels.
    pub height: u32,
    /// Linear RGBA radiance, `width * height * 4` floats in row-major order. The
    /// alpha channel is scene coverage: 0.0 on background pixels (matching the
    /// HDR scene clear), > 0.0 where geometry or the skybox was drawn.
    pub rgba: Vec<f32>,
}

impl ViewportRenderer {
    /// Render the lit scene from `camera` to an offscreen HDR target and return
    /// the linear radiance.
    ///
    /// The full HDR pipeline runs (shadows, transparency, skybox, IBL), so the
    /// captured lighting matches an on-screen frame. `size` is both width and
    /// height in pixels; a capture is always square, since its purpose is one
    /// face of an environment probe.
    ///
    /// `frame` supplies the scene, lighting, and environment. Its camera,
    /// viewport size, pixels-per-point, post-process toggle, and SSAA factor are
    /// overridden for the capture and restored before returning, so the caller's
    /// `FrameData` is left unchanged. The HDR path is forced on regardless of the
    /// caller's setting, because that is what raises the shader's `lit_clamp` to
    /// the `f16` maximum and lets radiance above 1.0 reach the `Rgba16Float`
    /// target this reads back.
    pub fn capture_hdr(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        camera: RenderCamera,
        size: u32,
    ) -> CapturedHdr {
        let size = size.max(1);

        // Snapshot every field the capture overrides so the caller's frame and
        // the renderer's dynamic-resolution state come back untouched.
        let saved_camera = std::mem::replace(&mut frame.camera.render_camera, camera);
        let saved_viewport_size = frame.camera.viewport_size;
        let saved_ppp = frame.camera.pixels_per_point;
        let saved_pp_enabled = frame.effects.post_process.enabled;
        let saved_ssaa = frame.effects.post_process.ssaa_factor;
        let saved_scale = self.current_render_scale;

        // Force a clean, full-resolution HDR render. The HDR path is what raises
        // lit_clamp to f16::MAX, so radiance stays unclamped into the
        // Rgba16Float scene target. SSAA and dynamic resolution are off so the
        // hdr_texture is exactly `size` x `size` and reads back one-to-one.
        frame.camera.viewport_size = [size as f32, size as f32];
        frame.camera.pixels_per_point = 1.0;
        frame.effects.post_process.enabled = true;
        frame.effects.post_process.ssaa_factor = 1;
        self.current_render_scale = 1.0;

        // A throwaway output view for the tone-map pass, which we do not read.
        let target_format = self.resources.target_format;
        let dummy = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("capture_dummy_output"),
            size: crate::gpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: target_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&crate::gpu::TextureViewDescriptor::default());

        // Render the full HDR frame; the pre-tonemap radiance lands in the
        // per-viewport hdr_texture.
        let cmd = self.render(device, queue, &dummy_view, frame);
        queue.submit(std::iter::once(cmd));

        // Read the HDR scene texture back before restoring anything.
        let vp_idx = frame.camera.viewport_index;
        let rgba = {
            let slot = &self.viewport_slots[vp_idx];
            let hdr = slot
                .hdr
                .as_ref()
                .expect("HDR state exists after an HDR render");
            Self::readback_rgba16f(device, queue, &hdr.hdr_texture, size, size)
        };

        // Restore the caller's frame and renderer state.
        frame.camera.render_camera = saved_camera;
        frame.camera.viewport_size = saved_viewport_size;
        frame.camera.pixels_per_point = saved_ppp;
        frame.effects.post_process.enabled = saved_pp_enabled;
        frame.effects.post_process.ssaa_factor = saved_ssaa;
        self.current_render_scale = saved_scale;

        CapturedHdr {
            width: size,
            height: size,
            rgba,
        }
    }

    /// Capture the environment radiance around `position` as an equirectangular
    /// panorama.
    ///
    /// Renders six 90-degree faces from `position` (each via
    /// [`capture_hdr`](Self::capture_hdr), so each face carries full scene
    /// lighting) and resolves them on the CPU into an equirect panorama that is
    /// `2 * equirect_height` wide by `equirect_height` tall. The direction ->
    /// UV mapping matches `dir_to_equirect_uv` in `helpers/ambient.wgsl`, so the
    /// result samples identically to an environment loaded from a file and can
    /// be handed straight to the IBL prefilter.
    ///
    /// `face_size` is the per-face render resolution; `frame` supplies the scene
    /// and lighting and is restored on return. Near/far planes are taken from
    /// the frame's current camera so the probe sees the same depth range as the
    /// on-screen view.
    pub fn capture_equirect(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        position: [f32; 3],
        face_size: u32,
        equirect_height: u32,
    ) -> CapturedHdr {
        let face_size = face_size.max(1);
        let eq_h = equirect_height.max(1);
        let eq_w = eq_h * 2;
        let eye = glam::Vec3::from(position);

        let near = frame.camera.render_camera.near.max(1e-3);
        let far = frame.camera.render_camera.far.max(near * 2.0);
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);

        // Six axis-aligned faces. Up vectors avoid being parallel to the view
        // direction; the exact choice is irrelevant to correctness because the
        // resolve re-projects each direction through the same face matrix used
        // to render it.
        let faces: [(glam::Vec3, glam::Vec3); 6] = [
            (glam::Vec3::X, glam::Vec3::Z),
            (glam::Vec3::NEG_X, glam::Vec3::Z),
            (glam::Vec3::Y, glam::Vec3::Z),
            (glam::Vec3::NEG_Y, glam::Vec3::Z),
            (glam::Vec3::Z, glam::Vec3::Y),
            (glam::Vec3::NEG_Z, glam::Vec3::Y),
        ];

        // (face pixels, forward direction, view_proj) per face.
        let mut captured: Vec<(CapturedHdr, glam::Vec3, glam::Mat4)> = Vec::with_capacity(6);
        for (dir, up) in faces {
            let mut cam = RenderCamera::default();
            cam.view = glam::Mat4::look_at_rh(eye, eye + dir, up);
            cam.projection = proj;
            cam.eye_position = position;
            cam.forward = dir.to_array();
            cam.near = near;
            cam.far = far;
            cam.fov = std::f32::consts::FRAC_PI_2;
            cam.aspect = 1.0;
            let view_proj = cam.view_proj();
            let face = self.capture_hdr(device, queue, frame, cam, face_size);
            captured.push((face, dir, view_proj));
        }

        // Resolve to equirect. For each output texel, form the world direction
        // per the consumer's convention (phi = atan2(y, x) around Z, theta =
        // asin(z)), pick the face whose forward is nearest that direction, and
        // bilinearly sample it at the direction's projected face UV.
        let mut rgba = vec![0.0f32; (eq_w * eq_h * 4) as usize];
        for y in 0..eq_h {
            let v = (y as f32 + 0.5) / eq_h as f32;
            let theta = (0.5 - v) * std::f32::consts::PI;
            let (st, ct) = theta.sin_cos();
            for x in 0..eq_w {
                let u = (x as f32 + 0.5) / eq_w as f32;
                let phi = (u - 0.5) * std::f32::consts::TAU;
                let (sp, cp) = phi.sin_cos();
                let d = glam::Vec3::new(ct * cp, ct * sp, st);

                let mut best = 0usize;
                let mut best_dot = f32::NEG_INFINITY;
                for (i, (_, fdir, _)) in captured.iter().enumerate() {
                    let dp = d.dot(*fdir);
                    if dp > best_dot {
                        best_dot = dp;
                        best = i;
                    }
                }

                let (face, _, view_proj) = &captured[best];
                let clip = *view_proj * (eye + d).extend(1.0);
                // clip.w is positive for points in front of the face camera.
                let inv_w = 1.0 / clip.w.max(1e-6);
                let fu = clip.x * inv_w * 0.5 + 0.5;
                let fv = 1.0 - (clip.y * inv_w * 0.5 + 0.5);
                let c = sample_face_bilinear(face, fu, fv);

                let o = ((y * eq_w + x) * 4) as usize;
                rgba[o] = c[0];
                rgba[o + 1] = c[1];
                rgba[o + 2] = c[2];
                rgba[o + 3] = c[3];
            }
        }

        CapturedHdr {
            width: eq_w,
            height: eq_h,
            rgba,
        }
    }

    /// Upload an SH light-probe field for dynamic objects to sample.
    ///
    /// Objects with [`IndirectLightSource::LightProbe`](crate::IndirectLightSource::LightProbe)
    /// take their indirect diffuse light from this field, blended at the object's
    /// position, instead of the global environment. Replaces any previous set.
    /// Sampled by every lit mesh path: opaque and transparent, per-object and
    /// instanced.
    pub fn set_light_probes(&mut self, probes: crate::resources::LightProbeSet) {
        self.resources.light_probes = Some(probes);
    }

    /// Remove the uploaded light-probe field. Objects revert to global IBL /
    /// hemisphere ambient.
    pub fn clear_light_probes(&mut self) {
        self.resources.light_probes = None;
    }

    /// Bake SH light probes at the given world positions.
    ///
    /// Captures an equirect panorama at each position with
    /// [`capture_equirect`](Self::capture_equirect) and projects it to order-2
    /// spherical harmonics, returning a [`LightProbeSet`] ready to sample. This
    /// is the generation half of the light-probe feature: run it once (at bake
    /// time, off the render loop) and keep the result to light dynamic objects
    /// by position.
    ///
    /// `face_size` is the per-probe cube-face resolution and `equirect_height`
    /// the projected panorama height; both trade capture time for angular
    /// accuracy of the low-frequency SH, so modest values (e.g. 64 / 64) are
    /// usually enough. `frame` is restored on return.
    pub fn bake_light_probes(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        positions: &[[f32; 3]],
        face_size: u32,
        equirect_height: u32,
    ) -> crate::resources::LightProbeSet {
        let probes = positions
            .iter()
            .map(|&position| {
                let panorama = self.capture_equirect(
                    device,
                    queue,
                    frame,
                    position,
                    face_size,
                    equirect_height,
                );
                let sh = crate::resources::project_equirect_to_sh(
                    &panorama.rgba,
                    panorama.width,
                    panorama.height,
                );
                crate::resources::LightProbe { position, sh }
            })
            .collect();
        crate::resources::LightProbeSet::new(probes)
    }

    /// Copy an `Rgba16Float` texture to the CPU and decode it to `f32` RGBA,
    /// stripping the 256-byte row padding wgpu requires on the copy.
    fn readback_rgba16f(
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        texture: &crate::gpu::Texture,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let bytes_per_pixel = 8u32; // Rgba16Float: four 16-bit channels.
        let unpadded_row = width * bytes_per_pixel;
        let align = crate::gpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);
        let buffer_size = (padded_row * height) as u64;

        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("capture_readback_staging"),
            size: buffer_size,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("capture_readback_encoder"),
        });
        encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            crate::gpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(crate::gpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(crate::gpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let _ = rx.recv().unwrap_or(Err(crate::gpu::BufferAsyncError));

        let mut out: Vec<f32> = Vec::with_capacity((width * height * 4) as usize);
        {
            let mapped = staging.slice(..).get_mapped_range();
            let data: &[u8] = &mapped;
            for row in 0..height as usize {
                let start = row * padded_row as usize;
                let row_bytes = &data[start..start + unpadded_row as usize];
                for texel in row_bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([texel[0], texel[1]]);
                    out.push(half::f16::from_bits(bits).to_f32());
                }
            }
        }
        staging.unmap();
        out
    }
}

/// Bilinearly sample a captured face's RGBA at `(u, v)` in `[0, 1]` (v is
/// top-down, matching the readback row order). UVs are clamped to the edge, so
/// directions that project just past a 90-degree face boundary sample the border
/// texel instead of wrapping.
fn sample_face_bilinear(face: &CapturedHdr, u: f32, v: f32) -> [f32; 4] {
    let w = face.width as usize;
    let h = face.height as usize;
    // Map to texel-center space and clamp.
    let fx = (u.clamp(0.0, 1.0) * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (v.clamp(0.0, 1.0) * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let texel = |x: usize, y: usize| -> [f32; 4] {
        let o = (y * w + x) * 4;
        [
            face.rgba[o],
            face.rgba[o + 1],
            face.rgba[o + 2],
            face.rgba[o + 3],
        ]
    };
    let c00 = texel(x0, y0);
    let c10 = texel(x1, y0);
    let c01 = texel(x0, y1);
    let c11 = texel(x1, y1);

    let mut out = [0.0f32; 4];
    for k in 0..4 {
        let top = c00[k] * (1.0 - tx) + c10[k] * tx;
        let bot = c01[k] * (1.0 - tx) + c11[k] * tx;
        out[k] = top * (1.0 - ty) + bot * ty;
    }
    out
}
