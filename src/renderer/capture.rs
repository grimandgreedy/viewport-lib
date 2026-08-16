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
use crate::gpu::util::DeviceExt;

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

/// A capture that stays on the GPU: an `Rgba16Float` texture holding linear HDR
/// radiance, with no CPU read back. This is the on-GPU counterpart of
/// [`CapturedHdr`], produced by [`ViewportRenderer::capture_hdr_gpu`] and
/// [`ViewportRenderer::capture_equirect_gpu`] for a capture -> resolve ->
/// prefilter path that never leaves the GPU. Read it to the CPU with
/// [`ViewportRenderer::read_captured_hdr`] when float pixels are wanted.
#[derive(Debug)]
pub struct CapturedHdrGpu {
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
    /// The linear HDR radiance texture (`Rgba16Float`, one mip, `COPY_SRC` and
    /// `TEXTURE_BINDING`). For a six-face capture this is the resolved equirect
    /// panorama, `2 * height` wide by `height` tall.
    pub texture: crate::gpu::Texture,
    /// A default 2D view of `texture`, ready to bind or sample.
    pub view: crate::gpu::TextureView,
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
        self.render_capture_frame(device, queue, frame, camera, size);

        // Read the HDR scene texture back from the per-viewport hdr_texture the
        // render left it in.
        let vp_idx = frame.camera.viewport_index;
        let rgba = {
            let slot = &self.viewport_slots[vp_idx];
            let hdr = slot
                .hdr
                .as_ref()
                .expect("HDR state exists after an HDR render");
            Self::readback_rgba16f(device, queue, &hdr.hdr_texture, size, size)
        };

        CapturedHdr {
            width: size,
            height: size,
            rgba,
        }
    }

    /// Render one full HDR frame from `camera` at `size` x `size` into the
    /// per-viewport `hdr_texture`, restoring the caller's frame and the
    /// renderer's dynamic-resolution state before returning.
    ///
    /// Both the CPU readback ([`capture_hdr`](Self::capture_hdr)) and the
    /// on-GPU copy path build on this: the pre-tonemap radiance is left in the
    /// viewport's `hdr_texture`, which the caller reads back or copies out
    /// before the next capture overwrites it.
    fn render_capture_frame(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        camera: RenderCamera,
        size: u32,
    ) {
        // Snapshot every field the capture overrides so the caller's frame and
        // the renderer's dynamic-resolution state come back untouched.
        let saved_camera = std::mem::replace(&mut frame.camera.render_camera, camera);
        let saved_viewport_size = frame.camera.viewport_size;
        let saved_ppp = frame.camera.pixels_per_point;
        let saved_pp_enabled = frame.effects.post_process.enabled;
        let saved_ssaa = frame.effects.post_process.ssaa_factor;
        let saved_scale = self.current_render_scale;

        // Mark this an auxiliary render for its duration: it reads the resident
        // scene and must not advance shared per-frame state. `render_mode`
        // suppresses the upload pump, the frame-counter bump, the HiZ prev-depth
        // store, and item-type plugins' prepare / cull; `last_stats` is a
        // multi-site value, so snapshot and restore it so the caller's
        // `last_frame_stats()` keeps reflecting their presented frame.
        let saved_render_mode = self.render_mode;
        let saved_last_stats = self.last_stats;
        self.render_mode = super::RenderMode::Derivative;

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

        // Derivative mode suppresses the upload pump, but sync mesh uploads defer
        // their slab geometry write to `process_uploads`; a capture can run
        // without an intervening presented frame, so flush geometry here or the
        // scene renders from unwritten (degenerate) vertices.
        self.resources.geometry.flush(queue);

        // Render the full HDR frame; the pre-tonemap radiance lands in the
        // per-viewport hdr_texture.
        let cmd = self.render(device, queue, &dummy_view, frame);
        queue.submit(std::iter::once(cmd));

        // Restore the caller's frame and renderer state. The hdr_texture is
        // untouched by this and holds the rendered radiance until the next
        // capture.
        frame.camera.render_camera = saved_camera;
        frame.camera.viewport_size = saved_viewport_size;
        frame.camera.pixels_per_point = saved_ppp;
        frame.effects.post_process.enabled = saved_pp_enabled;
        frame.effects.post_process.ssaa_factor = saved_ssaa;
        self.current_render_scale = saved_scale;
        self.render_mode = saved_render_mode;
        self.last_stats = saved_last_stats;
    }

    /// Render one HDR face and copy it into `dst_layer` of `dst` (an
    /// `Rgba16Float` texture of `size` x `size`), keeping the result on the GPU
    /// instead of reading it back to the CPU.
    fn copy_capture_face(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        camera: RenderCamera,
        size: u32,
        dst: &crate::gpu::Texture,
        dst_layer: u32,
    ) {
        self.render_capture_frame(device, queue, frame, camera, size);

        let vp_idx = frame.camera.viewport_index;
        let slot = &self.viewport_slots[vp_idx];
        let hdr = slot
            .hdr
            .as_ref()
            .expect("HDR state exists after an HDR render");

        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("capture_face_copy"),
        });
        encoder.copy_texture_to_texture(
            crate::gpu::TexelCopyTextureInfo {
                texture: &hdr.hdr_texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d::ZERO,
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin: crate::gpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: dst_layer,
                },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
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
    ///
    /// Like all captures, this reads the currently resident scene (auxiliary
    /// `Derivative` renders that do not advance the upload pipeline); bake once
    /// the scene is resident. See [`bake_light_probes`](Self::bake_light_probes).
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

    /// Render the lit scene from `camera` to an offscreen HDR target and return
    /// it as a GPU texture, without reading it back to the CPU.
    ///
    /// The on-GPU counterpart of [`capture_hdr`](Self::capture_hdr): same full
    /// HDR pipeline, same square `size` x `size` face, but the linear radiance
    /// stays in a returned [`CapturedHdrGpu`] texture. Use this when the next
    /// step is another GPU pass (the equirect resolve, a prefilter) and a CPU
    /// round trip would be wasted. `frame` is restored on return.
    pub fn capture_hdr_gpu(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        camera: RenderCamera,
        size: u32,
    ) -> CapturedHdrGpu {
        let size = size.max(1);
        let texture = new_hdr_target(device, "capture_hdr_gpu", size, size, 1, false);
        self.copy_capture_face(device, queue, frame, camera, size, &texture, 0);
        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        CapturedHdrGpu {
            width: size,
            height: size,
            texture,
            view,
        }
    }

    /// Capture the environment radiance around `position` as an equirectangular
    /// panorama, resolved on the GPU into a returned texture.
    ///
    /// The on-GPU counterpart of [`capture_equirect`](Self::capture_equirect):
    /// it renders the same six 90-degree faces from `position`, but keeps them
    /// in a GPU texture array and resolves them to equirect with a fragment
    /// pass ([`cube_to_equirect.wgsl`]) instead of a CPU loop. The direction ->
    /// UV mapping is identical, so the result samples the same as the CPU path
    /// and as an environment loaded from a file. The panorama is
    /// `2 * equirect_height` wide by `equirect_height` tall.
    ///
    /// `face_size` is the per-face render resolution; `frame` supplies the scene
    /// and lighting and is restored on return. Near/far planes come from the
    /// frame's current camera, matching `capture_equirect`.
    pub fn capture_equirect_gpu(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        position: [f32; 3],
        face_size: u32,
        equirect_height: u32,
    ) -> CapturedHdrGpu {
        let face_size = face_size.max(1);
        let eq_h = equirect_height.max(1);
        let eq_w = eq_h * 2;
        let eye = glam::Vec3::from(position);

        let near = frame.camera.render_camera.near.max(1e-3);
        let far = frame.camera.render_camera.far.max(near * 2.0);
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, near, far);

        // Same six axis-aligned faces and up vectors as the CPU resolve, so the
        // per-face projection convention matches exactly.
        let faces: [(glam::Vec3, glam::Vec3); 6] = [
            (glam::Vec3::X, glam::Vec3::Z),
            (glam::Vec3::NEG_X, glam::Vec3::Z),
            (glam::Vec3::Y, glam::Vec3::Z),
            (glam::Vec3::NEG_Y, glam::Vec3::Z),
            (glam::Vec3::Z, glam::Vec3::Y),
            (glam::Vec3::NEG_Z, glam::Vec3::Y),
        ];

        // Six-layer face array, one layer rendered per face.
        let face_array = new_hdr_target(device, "capture_faces", face_size, face_size, 6, false);

        let mut uniform = ResolveUniform::zeroed();
        uniform.eye = [eye.x, eye.y, eye.z, 0.0];
        for (i, (dir, up)) in faces.into_iter().enumerate() {
            let mut cam = RenderCamera::default();
            cam.view = glam::Mat4::look_at_rh(eye, eye + dir, up);
            cam.projection = proj;
            cam.eye_position = position;
            cam.forward = dir.to_array();
            cam.near = near;
            cam.far = far;
            cam.fov = std::f32::consts::FRAC_PI_2;
            cam.aspect = 1.0;
            uniform.view_proj[i] = cam.view_proj().to_cols_array_2d();
            uniform.forward[i] = [dir.x, dir.y, dir.z, 0.0];
            self.copy_capture_face(device, queue, frame, cam, face_size, &face_array, i as u32);
        }

        let texture = resolve_faces_to_equirect(device, queue, &face_array, &uniform, eq_w, eq_h);
        let view = texture.create_view(&crate::gpu::TextureViewDescriptor::default());
        CapturedHdrGpu {
            width: eq_w,
            height: eq_h,
            texture,
            view,
        }
    }

    /// Read a GPU capture back to CPU `f32` RGBA, the same layout
    /// [`CapturedHdr`] carries. Useful to inspect or feed the CPU `&[f32]` IBL
    /// entry once an on-GPU capture is in hand.
    pub fn read_captured_hdr(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        capture: &CapturedHdrGpu,
    ) -> CapturedHdr {
        let rgba = Self::readback_rgba16f(
            device,
            queue,
            &capture.texture,
            capture.width,
            capture.height,
        );
        CapturedHdr {
            width: capture.width,
            height: capture.height,
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
    ///
    /// This reads the *currently resident* scene: it renders auxiliary
    /// (`Derivative`) frames that do not advance the upload pipeline or the
    /// presented frame's state, so a streaming consumer should bake only once the
    /// geometry it wants captured is resident (gate on
    /// [`frame_fully_resident`](Self::frame_fully_resident) or
    /// `uploads_pending() == 0`). Meshes still streaming in are simply absent from
    /// the bake, not partially captured. Item-type plugin geometry is not included
    /// in the capture.
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

    /// Upload an adaptive probe volume for dynamic objects to sample per fragment.
    ///
    /// Objects with
    /// [`IndirectLightSource::ProbeVolume`](crate::IndirectLightSource::ProbeVolume)
    /// take their indirect diffuse from this volume, trilinearly interpolated at
    /// each shaded point, instead of the global environment. Replaces any
    /// previous volume and rebuilds the camera bind groups.
    pub fn set_light_probe_volume(
        &mut self,
        device: &crate::gpu::Device,
        volume: &crate::resources::LightProbeVolume,
    ) {
        let gpu = volume.to_gpu();
        self.resources.light_probe_volume_buf = Some(device.create_buffer_init(
            &crate::gpu::util::BufferInitDescriptor {
                label: Some("light_probe_volume_buf"),
                contents: bytemuck::cast_slice(&gpu),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            },
        ));
        self.rebuild_camera_bind_groups(device);
    }

    /// Remove the uploaded probe volume. Objects that opted into it fall back to
    /// no indirect (a disabled header is bound in its place).
    pub fn clear_light_probe_volume(&mut self, device: &crate::gpu::Device) {
        self.resources.light_probe_volume_buf = None;
        self.rebuild_camera_bind_groups(device);
    }

    /// Bake an adaptive probe volume: a regular `dims` grid of SH probes over the
    /// box starting at `min` with `cell_size` spacing.
    ///
    /// Captures an equirect panorama at each grid position with
    /// [`capture_equirect`](Self::capture_equirect) and projects it to order-2
    /// SH, returning a [`LightProbeVolume`](crate::resources::LightProbeVolume)
    /// ready to upload with [`set_light_probe_volume`](Self::set_light_probe_volume).
    /// This is the generation half: run it once at bake time. `face_size` and
    /// `equirect_height` trade capture time for angular accuracy; `frame` is
    /// restored on return.
    pub fn bake_light_probe_volume(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        min: [f32; 3],
        cell_size: [f32; 3],
        dims: [u32; 3],
        face_size: u32,
        equirect_height: u32,
    ) -> crate::resources::LightProbeVolume {
        let dims = [dims[0].max(1), dims[1].max(1), dims[2].max(1)];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let mut sh = Vec::with_capacity(count);
        for iz in 0..dims[2] {
            for iy in 0..dims[1] {
                for ix in 0..dims[0] {
                    let position = [
                        min[0] + ix as f32 * cell_size[0],
                        min[1] + iy as f32 * cell_size[1],
                        min[2] + iz as f32 * cell_size[2],
                    ];
                    let panorama = self.capture_equirect(
                        device,
                        queue,
                        frame,
                        position,
                        face_size,
                        equirect_height,
                    );
                    sh.push(crate::resources::project_equirect_to_sh(
                        &panorama.rgba,
                        panorama.width,
                        panorama.height,
                    ));
                }
            }
        }
        crate::resources::LightProbeVolume::new(min, cell_size, dims, sh)
    }

    /// Bake a reflection probe at the centre of `bounds` and return a
    /// parallax-enabled [`EnvironmentZone`](crate::resources::EnvironmentZone) for it.
    ///
    /// Captures the scene radiance around the box centre with
    /// [`capture_equirect`](Self::capture_equirect), prefilters it into a fresh
    /// environment layer, and returns a zone that selects that layer inside
    /// `bounds` and box-projects reflections against it. Add the zone to the set
    /// with [`set_environment_zones`](Self::set_environment_zones). Blocks until
    /// the bake finishes and rebuilds the camera bind groups.
    ///
    /// `face_size` is the per-face capture resolution and `equirect_height` the
    /// projected panorama height; `frame` supplies the scene and is restored on
    /// return. To bake several probes, prefer
    /// [`capture_reflection_probes`](Self::capture_reflection_probes), which
    /// rebuilds the bind groups once instead of per probe.
    pub fn capture_reflection_probe(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        bounds: crate::scene::aabb::Aabb,
        fade_distance: f32,
        face_size: u32,
        equirect_height: u32,
    ) -> crate::error::ViewportResult<crate::resources::EnvironmentZone> {
        let zones = self.capture_reflection_probes(
            device,
            queue,
            frame,
            &[(bounds, fade_distance)],
            face_size,
            equirect_height,
        )?;
        Ok(zones[0])
    }

    /// Bake several reflection probes in one pass, rebuilding the camera bind
    /// groups once at the end rather than once per probe.
    ///
    /// Each entry is `(bounds, fade_distance)`; the probe is captured at the box
    /// centre. Returns one parallax-enabled
    /// [`EnvironmentZone`](crate::resources::EnvironmentZone) per entry, in order,
    /// ready to hand to [`set_environment_zones`](Self::set_environment_zones).
    /// Probes are captured under the current scene lighting (the default
    /// environment), not each other, so this is a single-bounce bake.
    pub fn capture_reflection_probes(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &mut FrameData,
        probes: &[(crate::scene::aabb::Aabb, f32)],
        face_size: u32,
        equirect_height: u32,
    ) -> crate::error::ViewportResult<Vec<crate::resources::EnvironmentZone>> {
        let mut zones = Vec::with_capacity(probes.len());
        for &(bounds, fade_distance) in probes {
            let center = bounds.center().to_array();
            let panorama =
                self.capture_equirect(device, queue, frame, center, face_size, equirect_height);
            // Resources-level upload does not rebuild bind groups; we rebuild once
            // after the whole batch below.
            let environment = crate::resources::material::environment::upload_environment(
                &mut self.resources,
                device,
                queue,
                &panorama.rgba,
                panorama.width,
                panorama.height,
            )?;
            zones.push(crate::resources::EnvironmentZone {
                bounds,
                environment,
                fade_distance,
                parallax: true,
            });
        }
        self.rebuild_camera_bind_groups(device);
        Ok(zones)
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

/// Create an `Rgba16Float` render/sample target. `layers` > 1 makes a 2D array
/// (one layer per cube face). Usage always allows sampling and copies; the
/// resolve output additionally needs `RENDER_ATTACHMENT`, requested via `render`.
fn new_hdr_target(
    device: &crate::gpu::Device,
    label: &str,
    width: u32,
    height: u32,
    layers: u32,
    render: bool,
) -> crate::gpu::Texture {
    let mut usage = crate::gpu::TextureUsages::TEXTURE_BINDING
        | crate::gpu::TextureUsages::COPY_DST
        | crate::gpu::TextureUsages::COPY_SRC;
    if render {
        usage |= crate::gpu::TextureUsages::RENDER_ATTACHMENT;
    }
    device.create_texture(&crate::gpu::TextureDescriptor {
        label: Some(label),
        size: crate::gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: crate::gpu::TextureDimension::D2,
        format: crate::gpu::TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    })
}

/// The per-face matrices and eye the GPU resolve reads. Mirrors the `Resolve`
/// uniform in `cube_to_equirect.wgsl`: six view-projection matrices, six
/// forward directions, and the capture eye.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveUniform {
    view_proj: [[[f32; 4]; 4]; 6],
    forward: [[f32; 4]; 6],
    eye: [f32; 4],
}

impl ResolveUniform {
    fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

/// Resolve six captured faces (a 6-layer `Rgba16Float` array) into an equirect
/// panorama with a fullscreen fragment pass. The GPU counterpart of the CPU
/// loop in [`ViewportRenderer::capture_equirect`].
fn resolve_faces_to_equirect(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    faces: &crate::gpu::Texture,
    uniform: &ResolveUniform,
    eq_w: u32,
    eq_h: u32,
) -> crate::gpu::Texture {
    let output = new_hdr_target(device, "capture_equirect_gpu", eq_w, eq_h, 1, true);
    let output_view = output.create_view(&crate::gpu::TextureViewDescriptor::default());

    let faces_view = faces.create_view(&crate::gpu::TextureViewDescriptor {
        dimension: Some(crate::gpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    // Clamp-to-edge on both axes: a face UV never wraps, unlike a full equirect.
    let sampler = device.create_sampler(&crate::gpu::SamplerDescriptor {
        label: Some("capture_resolve_sampler"),
        address_mode_u: crate::gpu::AddressMode::ClampToEdge,
        address_mode_v: crate::gpu::AddressMode::ClampToEdge,
        address_mode_w: crate::gpu::AddressMode::ClampToEdge,
        mag_filter: crate::gpu::FilterMode::Linear,
        min_filter: crate::gpu::FilterMode::Linear,
        ..Default::default()
    });

    let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
        label: Some("capture_resolve_uniform"),
        contents: bytemuck::bytes_of(uniform),
        usage: crate::gpu::BufferUsages::UNIFORM,
    });

    let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("capture_resolve_bgl"),
        entries: &[
            crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Texture {
                    sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                    view_dimension: crate::gpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            crate::gpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                count: None,
            },
            crate::gpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some("capture_resolve_bg"),
        layout: &bgl,
        entries: &[
            crate::gpu::BindGroupEntry {
                binding: 0,
                resource: crate::gpu::BindingResource::TextureView(&faces_view),
            },
            crate::gpu::BindGroupEntry {
                binding: 1,
                resource: crate::gpu::BindingResource::Sampler(&sampler),
            },
            crate::gpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    });

    let module = crate::resources::builders::wgsl_module(
        device,
        "cube_to_equirect",
        crate::resources::builders::wgsl_source!("cube_to_equirect"),
    );
    let layout =
        crate::resources::builders::pipeline_layout(device, "capture_resolve_layout", &[&bgl]);
    let pipeline = crate::resources::builders::build_fullscreen_pipeline(
        device,
        "capture_resolve_pipeline",
        &layout,
        &module,
        crate::gpu::TextureFormat::Rgba16Float,
        None,
    );

    let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
        label: Some("capture_resolve_encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
            #[cfg(feature = "wgpu29")]
            multiview_mask: None,
            label: Some("capture_resolve_pass"),
            color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                view: &output_view,
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
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(std::iter::once(encoder.finish()));

    output
}

#[cfg(test)]
mod tests {
    use crate::camera::Camera;
    use crate::renderer::types::FrameData;
    use crate::renderer::{
        CameraFrame, RenderCamera, SceneFrame, SurfaceSubmission, ViewportRenderer,
    };
    use crate::resources::UploadStatus;

    fn headless_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
                label: Some("capture_tests"),
                ..Default::default()
            }))
            .ok()?;
        Some((device, queue))
    }

    fn empty_frame() -> FrameData {
        let render_cam = RenderCamera::from_camera(&Camera::default());
        let cf = CameraFrame::new(render_cam, [64.0, 64.0]);
        let sf = SceneFrame::new(SurfaceSubmission::Flat(std::sync::Arc::from(Vec::new())));
        FrameData::new(cf, sf)
    }

    // A derivative render (probe bake) must not pump the upload pipeline. If it
    // did, it would promote and then reap the one-cycle promotion window a
    // consumer polls, stranding a deferred bind permanently.
    #[test]
    fn capture_does_not_strand_a_pending_upload() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping capture_does_not_strand_a_pending_upload: no GPU adapter");
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

        let job = renderer
            .resources_mut()
            .begin_upload_mesh_data(&device, crate::primitives::cube(1.0))
            .unwrap();
        assert!(
            matches!(renderer.upload_status(job), UploadStatus::Pending { .. }),
            "upload should be in flight before any presented drain"
        );

        // Bake before the upload promotes. With the derivative-render fix the
        // capture leaves the pending job untouched.
        let mut fd = empty_frame();
        let _ = renderer.bake_light_probes(&device, &queue, &mut fd, &[[0.0, 0.0, 0.0]], 8, 8);
        assert!(
            matches!(renderer.upload_status(job), UploadStatus::Pending { .. }),
            "capture must not advance or reap the pending upload"
        );

        // A presented drain still promotes it to Ready.
        for _ in 0..16 {
            renderer.resources_mut().process_uploads(&device, &queue);
            if matches!(renderer.upload_status(job), UploadStatus::Ready) {
                break;
            }
        }
        assert!(matches!(renderer.upload_status(job), UploadStatus::Ready));
    }

    // The reflection-probe bake uploads an environment per probe, which blocks
    // on a runner drain. That drain must not clear the promotion window and
    // strand a streaming consumer's in-flight mesh upload.
    #[test]
    fn reflection_bake_does_not_strand_a_pending_upload() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping reflection_bake_does_not_strand_a_pending_upload: no GPU adapter");
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

        let job = renderer
            .resources_mut()
            .begin_upload_mesh_data(&device, crate::primitives::cube(1.0))
            .unwrap();
        assert!(matches!(
            renderer.upload_status(job),
            UploadStatus::Pending { .. }
        ));

        let mut fd = empty_frame();
        let bounds = crate::scene::aabb::Aabb {
            min: glam::Vec3::splat(-1.0),
            max: glam::Vec3::splat(1.0),
        };
        let _zone = renderer
            .capture_reflection_probe(&device, &queue, &mut fd, bounds, 1.0, 8, 8)
            .unwrap();

        // The bake's environment upload drained the runner, promoting the mesh,
        // but retained the promotion window, so the consumer's next poll still
        // observes `Ready` rather than a reaped `Unknown`. (Do not pump again
        // here: an ordinary clearing `process_uploads` is exactly what the
        // consumer's next presented prepare does, after they have observed it.)
        assert!(
            matches!(renderer.upload_status(job), UploadStatus::Ready),
            "reflection bake must promote and retain the pending upload, not strand it"
        );
    }

    // A derivative render must not advance the frame counter, which drives the
    // presented frame's temporal phase (scatter jitter, pick cadence).
    #[test]
    fn capture_does_not_advance_frame_counter() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping capture_does_not_advance_frame_counter: no GPU adapter");
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

        let mut fd = empty_frame();
        renderer.prepare(&device, &queue, &fd); // presented frame advances the counter
        let before = renderer.frame_counter;

        // Six face renders per probe; none should touch the counter.
        let _ = renderer.bake_light_probes(&device, &queue, &mut fd, &[[0.0, 0.0, 0.0]], 8, 8);
        assert_eq!(
            renderer.frame_counter, before,
            "capture must not advance the frame counter"
        );
    }

    // render_offscreen_snapshot is a derivative render: it must not pump the
    // upload pipeline, whereas the presented render_offscreen must.
    #[test]
    fn render_offscreen_snapshot_does_not_pump_uploads() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping render_offscreen_snapshot_does_not_pump_uploads: no GPU adapter");
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let job = renderer
            .resources_mut()
            .begin_upload_mesh_data(&device, crate::primitives::cube(1.0))
            .unwrap();

        let fd = empty_frame();
        let _ = renderer.render_offscreen_snapshot(&device, &queue, &fd, 64, 64);
        assert!(
            matches!(renderer.upload_status(job), UploadStatus::Pending { .. }),
            "snapshot offscreen render must not advance the upload pipeline"
        );

        // The presented offscreen render pumps it toward Ready.
        for _ in 0..16 {
            let _ = renderer.render_offscreen(&device, &queue, &fd, 64, 64);
            if matches!(renderer.upload_status(job), UploadStatus::Ready) {
                break;
            }
        }
        assert!(matches!(renderer.upload_status(job), UploadStatus::Ready));
    }

    // mesh_resident / frame_fully_resident are level queries over the store.
    #[test]
    fn residency_queries_track_the_mesh_store() {
        let Some((device, _queue)) = headless_device() else {
            eprintln!("skipping residency_queries_track_the_mesh_store: no GPU adapter");
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);

        let mesh = renderer
            .resources_mut()
            .upload_mesh_data(&device, &crate::primitives::cube(1.0))
            .unwrap();
        assert!(renderer.mesh_resident(mesh), "uploaded mesh is resident");

        // Empty frame: trivially fully resident.
        assert!(renderer.frame_fully_resident(&empty_frame()));

        // A frame referencing the resident mesh is fully resident.
        let item = crate::SceneRenderItem {
            mesh_id: mesh,
            ..Default::default()
        };
        let cf = CameraFrame::new(RenderCamera::from_camera(&Camera::default()), [64.0, 64.0]);
        let fd = FrameData::new(cf, SceneFrame::from_surface_items(vec![item]));
        assert!(renderer.frame_fully_resident(&fd));

        // Remove the mesh: both queries flip.
        assert!(renderer.resources_mut().remove_mesh(mesh));
        assert!(
            !renderer.mesh_resident(mesh),
            "removed mesh is not resident"
        );
        assert!(
            !renderer.frame_fully_resident(&fd),
            "a frame referencing a non-resident mesh is not fully resident"
        );
    }

    // Records how often each dispatched item-type plugin hook was called.
    #[derive(Default)]
    struct PluginCalls {
        prepare: std::sync::atomic::AtomicUsize,
        cull: std::sync::atomic::AtomicUsize,
        paint: std::sync::atomic::AtomicUsize,
    }

    struct MockPlugin {
        calls: std::sync::Arc<PluginCalls>,
    }

    impl crate::plugin_api::ItemTypePlugin for MockPlugin {
        fn type_name(&self) -> &'static str {
            "mock"
        }
        fn prepare(
            &mut self,
            _device: &crate::gpu::Device,
            _queue: &crate::gpu::Queue,
            _ctx: &crate::plugin_api::ItemFrameContext<'_>,
            _items: &dyn crate::plugin_api::PluginItemCollection,
        ) -> Vec<crate::gpu::CommandBuffer> {
            self.calls
                .prepare
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Vec::new()
        }
        fn cull(
            &mut self,
            _frustum: &crate::camera::frustum::Frustum,
            _ctx: &crate::plugin_api::ItemFrameContext<'_>,
            _items: &dyn crate::plugin_api::PluginItemCollection,
        ) {
            self.calls
                .cull
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn paint<'a>(
            &'a self,
            _pass: &mut crate::gpu::RenderPass<'a>,
            _ctx: &crate::plugin_api::PaintContext<'a>,
            _items: &'a dyn crate::plugin_api::PluginItemCollection,
        ) {
            self.calls
                .paint
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    struct MockItems {
        settings: crate::scene::material::ItemSettings,
    }

    impl crate::plugin_api::PluginItemCollection for MockItems {
        fn len(&self) -> usize {
            1
        }
        fn item_settings(&self, _index: usize) -> &crate::scene::material::ItemSettings {
            &self.settings
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    // A derivative render (bake) must not dispatch any item-type plugin hook:
    // not the `&mut self` prepare / cull (which advance plugin state), nor the
    // draw passes (which would render stale, wrong-camera geometry, since cull
    // was skipped). A presented render dispatches all three.
    #[test]
    fn bake_does_not_dispatch_item_type_plugins() {
        use std::sync::atomic::Ordering::Relaxed;
        let Some((device, queue)) = headless_device() else {
            eprintln!("skipping bake_does_not_dispatch_item_type_plugins: no GPU adapter");
            return;
        };
        let mut renderer =
            ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
        let calls = std::sync::Arc::new(PluginCalls::default());
        renderer.with_item_type_plugin(
            &device,
            Box::new(MockPlugin {
                calls: calls.clone(),
            }),
        );

        let mut fd = empty_frame();
        // Route through the HDR path so the opaque plugin paint dispatch runs.
        fd.effects.post_process.enabled = true;
        fd.scene.submit_plugin_items(
            "mock",
            MockItems {
                settings: crate::scene::material::ItemSettings {
                    pick_id: crate::renderer::PickId(1),
                    ..Default::default()
                },
            },
        );

        // A presented render dispatches prepare, cull, and paint.
        let target = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("mock_plugin_target"),
            size: crate::gpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Bgra8UnormSrgb,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&crate::gpu::TextureViewDescriptor::default());
        let cmd = renderer.render(&device, &queue, &view, &fd);
        queue.submit(std::iter::once(cmd));

        assert!(
            calls.prepare.load(Relaxed) >= 1,
            "presented render runs prepare"
        );
        assert!(calls.cull.load(Relaxed) >= 1, "presented render runs cull");
        assert!(
            calls.paint.load(Relaxed) >= 1,
            "presented render runs paint"
        );

        let (p, c, pt) = (
            calls.prepare.load(Relaxed),
            calls.cull.load(Relaxed),
            calls.paint.load(Relaxed),
        );

        // The bake is a derivative render: no plugin hook should fire.
        let _ = renderer.bake_light_probes(&device, &queue, &mut fd, &[[0.0, 0.0, 0.0]], 8, 8);
        assert_eq!(
            calls.prepare.load(Relaxed),
            p,
            "bake must not call plugin prepare"
        );
        assert_eq!(
            calls.cull.load(Relaxed),
            c,
            "bake must not call plugin cull"
        );
        assert_eq!(
            calls.paint.load(Relaxed),
            pt,
            "bake must not call plugin paint"
        );
    }
}
